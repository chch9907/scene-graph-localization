import json
import numpy as np
import distinctipy
from typing import Any, Dict, List, Sequence
import copy
import os
import pickle
import open_clip
import torch
import torch.nn as nn
import open3d as o3d
from conceptgraph.slam.mapping import compute_spatial_similarities, compute_visual_similarities
from localization_utils import softmax, minmax_norm, get_visible_areas, stack_pcd_bboxes
from sklearn.cluster import AgglomerativeClustering
from conceptgraph.utils.ious import (
    compute_iou_batch, 
    compute_giou_batch, 
    compute_3d_iou_accuracte_batch, 
    compute_3d_giou_accurate_batch,
)

def query_scene_graph_semantics(
    position: np.ndarray,
    scene_graph: Dict[str, Any],
    radius: float,
) -> set[str]:
    """Query semantic labels (legacy version, kept for compatibility)"""
    semantics: set[str] = set()
    nodes = scene_graph.get("nodes", [])
    for node in nodes:
        node_pos = node.get("position") or node.get("pose", {}).get("position")
        if node_pos is None:
            continue
        node_vec = np.array(
            [node_pos.get("x", 0.0), node_pos.get("y", 0.0), node_pos.get("z", 0.0)],
            dtype=np.float32,
        )
        if np.linalg.norm(node_vec - position) <= radius:
            category = node.get("category") or node.get("label") or node.get("objectType")
            if category:
                semantics.add(str(category))
    return semantics


def query_scene_graph_features(
    position: np.ndarray,
    scene_graph: Any,  # MapObjectList
    radius: float = 4.0,
    fov: float = 90.0,
) -> np.ndarray:
    """Query CLIP features of objects within radius/FOV of given pose.
    
    Args:
        position: Pose [x, y, z, yaw_deg]
        scene_graph: MapObjectList containing objects with 'bbox' and 'clip_ft'
        radius: search radius
    
    Returns:
        Array of CLIP features (N, feature_dim) where N is number of nearby objects
    """

    # restrict query to a circular sector aligned with the robot heading
    pose = np.asarray(position, dtype=np.float32)
    orientation = float(pose[3]) if pose.shape[0] > 3 else 0.0
    orientation_rad = np.deg2rad(orientation)
    forward_dir = np.array([np.cos(orientation_rad), np.sin(orientation_rad)], dtype=np.float32)
    half_fov_rad = np.deg2rad(fov * 0.5)
    pos_xyz = pose[:3]

    nearby_features = []
    nearby_obj_ids = []
    for idx, obj in enumerate(scene_graph):
        if obj.get('is_background', False):
            continue
        bbox = obj.get('bbox')
        if bbox is None:
            continue
        # roughly search nearby objects by distance 
        obj_center = np.array(bbox.center, dtype=np.float32)
        rel_vec = (obj_center - pos_xyz)[[0, 2]]
        distance = np.linalg.norm(rel_vec)
        if distance > radius:
            continue
        
        # check if within FOV to avoid matching objects behind the robot
        rel_norm = distance
        if rel_norm < 0.5:
            within_fov = True
        else:
            # obtain the angle between the robot's forward direction and the object direction
            rel_dir = rel_vec / rel_norm
            cos_angle = np.clip(np.dot(forward_dir, rel_dir), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            within_fov = angle <= half_fov_rad
        if not within_fov:
            continue

        clip_ft = obj.get('clip_ft')
        if clip_ft is not None:
            nearby_features.append(clip_ft)
            nearby_obj_ids.append(idx)
    
    if len(nearby_features) == 0:
        return np.array([], dtype=np.float32), [] 
    
    return np.array(nearby_features, dtype=np.float32), np.array(nearby_obj_ids)
    

class SemanticParticleFilter:
    def __init__(
        self,
        reachable_positions: Sequence[Dict[str, float]],
        scene_graph: Dict[str, Any],
        class_set: List[str],
        num_particles: int,
        motion_noise: float,
        rotation_noise: float,
        random_state: np.random.Generator,
        temperature: float=1.0,
        alpha:float=0.5,
        use_conf: bool=False,
        fov: float = 90.0,
        cluster_dist_thred: float = 6, #! careful
        max_vis_dist: float = 4,
        connectivity_distance: float = 1.5,
        init_particles_path: str | None = None,
        label_cache_path: str | None = None,
    ) -> None:
        self.scene_graph = scene_graph
        self.reachable = np.array(
            [[p["x"], p.get("y", 0.0), p["z"]] for p in reachable_positions],
            dtype=np.float32,
        )
        self.connectivity_distance = max(float(connectivity_distance), 1e-3)
        self.class_set = list(class_set)
        if label_cache_path:
            self.label_cache_path = (
                label_cache_path
                if label_cache_path.endswith('.json')
                else f"{label_cache_path}.json"
            )
        else:
            self.label_cache_path = None
        self.use_conf = use_conf

        # extract information from scene graph
        self.obj_positions, self.obj_volumes, self.obj_valid_indices = \
            self.get_scene_obj_position(scene_graph)
        self.visible_areas, self.visible_areas_sizes = get_visible_areas(self.obj_positions, 
                                                 self.reachable[:, [0, 2]], 
                                                 max_vis_dist)

        ## cluster objects based on positions
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_dist_thred,
            linkage="ward",
            )
        # Group nearby objects into pseudo-rooms so TF/IDF statistics remain spatially grounded
        self.cluster_labels = clustering.fit_predict(self.obj_positions[:, [0, 2]])  # only use x, z for clustering
        cluster_indices_dict = dict()
        centroids = dict()

        # get the centroid position of each cluster
        for clusters_id in set(self.cluster_labels):
            cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label == clusters_id]
            cluster_positions = [self.obj_positions[i] for i in cluster_indices]
            cluster_positions = np.array(cluster_positions)[:, [0, 2]]
            ori_centroid = np.mean(cluster_positions, axis=0)
            
            centroid = cluster_positions[
                np.argmin(np.linalg.norm(np.array(cluster_positions) - 
                                         ori_centroid
                                         , axis=1))
            ]
            centroids[clusters_id] = centroid
            cluster_indices_dict[clusters_id] = cluster_indices

        # compute object distance to cluster centroid
        # Track how far each object lies from the densest point of its cluster for the centrality cue
        self.obj_dist_to_centroid = []
        for obj_pos, clusters_id in zip(self.obj_positions, self.cluster_labels):
            dist_to_centroid = np.linalg.norm(obj_pos[[0, 2]] - centroids[clusters_id]) 
            self.obj_dist_to_centroid.append(dist_to_centroid)

        # measure object connectivity within each cluster using a distance-based graph
        self.obj_connectivity = np.zeros(len(self.obj_positions), dtype=np.float32)
        for cluster_id, cluster_indices in cluster_indices_dict.items():
            cluster_size = len(cluster_indices)
            if cluster_size == 0:
                continue
            if cluster_size == 1:
                self.obj_connectivity[cluster_indices[0]] = 0.0
                continue

            cluster_positions = self.obj_positions[cluster_indices][:, [0, 2]]  # only x, z
            diff = cluster_positions[:, None, :] - cluster_positions[None, :, :]
            dists = np.linalg.norm(diff, axis=2)

            # Build a simple proximity graph to approximate which objects co-occur in views
            adjacency = (dists <= self.connectivity_distance)
            np.fill_diagonal(adjacency, False)

            normalized_degree = adjacency.sum(axis=1).astype(np.float32) / (float(cluster_size) - 1)
            for local_idx, global_idx in enumerate(cluster_indices):
                self.obj_connectivity[global_idx] = normalized_degree[local_idx]

        
        # Reuse label predictions when possible to avoid expensive CLIP forward passes
        cached_labels = self._load_obj_labels_from_cache()
        if cached_labels is None:
            self.obj_labels = self._compute_obj_labels()
            self._save_obj_labels_cache()
        else:
            self.obj_labels = cached_labels
        

        # Keep a compact dictionary keyed by original scene-graph indices for fast lookups later
        self.processed_scene_graph = {valid_idx: {
            'pose': self.obj_positions[i],
            'volume': self.obj_volumes[i],
            'visible_area': self.visible_areas_sizes[i],
            'clip_ft': self.scene_graph[valid_idx]['clip_ft'],  # can have invalid indices
            'label': self.obj_labels[i],
            'cluster_id': self.cluster_labels[i],  
            'dist_to_centroid': self.obj_dist_to_centroid[i],
            'connectivity': self.obj_connectivity[i],
            } for i, valid_idx in enumerate(self.obj_valid_indices)
        }

        # information_gain u = TF-IDF + degrees + saliency        
        cluster_labels_dict = dict()
        for cluster_id, cluster_indices in cluster_indices_dict.items():
            cluster_i_labels = [self.obj_labels[j] for j in cluster_indices]
            cluster_labels_dict[cluster_id] = cluster_i_labels

        # compute TF
        obj_tf = list()
        for obj_label, cluster_id in zip(self.obj_labels, self.cluster_labels):
            cluster_labels_list = cluster_labels_dict[cluster_id]
            tf = cluster_labels_list.count(obj_label) / len(cluster_labels_list)
            obj_tf.append(tf)
        obj_tf_array = 1 / np.array(obj_tf)  # opposite of TF

        # compute IDF
        obj_labels_set = set(self.obj_labels)
        N_cluster = len(obj_labels_set)
        obj_idf = {label: np.log(N_cluster / (sum([1 for cluster_labels_list in cluster_labels_dict.values() 
                                                   if label in cluster_labels_list]))) 
                                                   for label in obj_labels_set}
        obj_idf_array = np.array([obj_idf[label] for label in self.obj_labels])

        # compute TF-IDF
        u_tf_idf = minmax_norm(obj_tf_array + obj_idf_array)  
   

        # compute cluster centrality based on connectivity
        u_connectivity = minmax_norm(np.array(self.obj_connectivity, dtype=np.float32))
        print('u_connectivity', u_connectivity)


      
        self.info_weights = alpha * u_tf_idf + (1-alpha) * u_connectivity
        # self.info_weights += 0.01 # add a small bias to avoid zero weights
        print('info_weights', self.info_weights)
        print('alpha', alpha)

        
        for obj_idx, weight in zip(self.obj_valid_indices, self.info_weights):
            self.processed_scene_graph[obj_idx]['info_weight'] = weight

        # set up particle filter parameters
        self.min_x = np.min(self.reachable[:, 0])
        self.max_x = np.max(self.reachable[:, 0])
        self.min_z = np.min(self.reachable[:, 2])
        self.max_z = np.max(self.reachable[:, 2])
        self.num_particles = num_particles
        self.motion_noise = motion_noise
        self.rotation_noise = rotation_noise
        self.rng = random_state
        self.temperature = max(temperature, 1e-3)
        if init_particles_path:
            self.init_particles_path = (
                init_particles_path
                if init_particles_path.endswith('.npy')
                else f"{init_particles_path}.npy"
            )
        else:
            self.init_particles_path = None
        self.particles = self._init_particles()
        self.fov = fov
        self.weights = np.ones(num_particles, dtype=np.float32) / num_particles


    def get_scene_obj_position(self, scene_graph):
        # Pre-filter objects with valid bounding boxes so downstream tensors stay compact
        obj_positions = list()  
        obj_volumes = list()
        obj_valid_indices = list()
        for idx, obj in enumerate(scene_graph):
            if obj.get('is_background', False):
                continue
            bbox = obj.get('bbox')
            if bbox is None:
                continue
            # Get bbox center position #TODO: consider orientation
            obj_center = np.array(bbox.center, dtype=np.float32)
            obj_positions.append(obj_center[:3])  # [0, 1, 2]
            obj_volumes.append(bbox.volume())
            obj_valid_indices.append(idx)
        return np.array(obj_positions), np.array(obj_volumes), np.array(obj_valid_indices)

    def get_pcd_bboxes(self, indices=None):
        all_bboxes = [obj['bbox'] for obj in self.scene_graph]
        if indices is not None:
            return [all_bboxes[i] for i in indices]
        return all_bboxes

    def _init_particles(self) -> np.ndarray:
        cache_path = self.init_particles_path
        if cache_path and os.path.exists(cache_path):
            try:
                particles = np.load(cache_path).astype(np.float32)
                if particles.shape == (self.num_particles, 4):
                    print(f"Loaded initial particles from {cache_path}")
                    return particles
                else:
                    print(
                        f"Cached particles at {cache_path} have shape {particles.shape}, "
                        f"expected {(self.num_particles, 4)}; resampling instead."
                    )
            except (OSError, ValueError) as exc:
                print(f"Failed to load initial particles from {cache_path}: {exc}; resampling instead.")
        
        idx = self.rng.choice(len(self.reachable), size=self.num_particles)
        particles = np.zeros((self.num_particles, 4), dtype=np.float32)  # x, y, z, rotation_y
        particles[:, :3] = self.reachable[idx]
        particles[:, 3] = self.rng.uniform(-180.0, 180.0, size=self.num_particles)

        # if cache_path:  # save particles to reproduce results at subsequent runs
        #     np.save(cache_path, particles)
        #     print(f"Saved initial particles to {cache_path}")

        return particles

    def _sample_motion_noise(self, trans_std: float, rot_std: float) -> tuple[np.ndarray, np.ndarray]:
        trans_noise = np.zeros((self.num_particles, 3), dtype=np.float32)
        rot_noise = np.zeros(self.num_particles, dtype=np.float32)
        if trans_std > 0:
            trans_noise = self.rng.normal(0.0, trans_std, size=(self.num_particles, 3)).astype(np.float32)
        if rot_std > 0:
            rot_noise = self.rng.normal(0.0, rot_std, size=self.num_particles).astype(np.float32)
        return trans_noise, rot_noise

    def predict(self, delta_translation: np.ndarray, delta_rotation: float, dist_thred = 1.0) -> None:
        trans_noise, rot_noise = self._sample_motion_noise(self.motion_noise, self.rotation_noise)
        self.particles[:, :3] += delta_translation + trans_noise
        self.particles[:, 3] += delta_rotation + rot_noise
        self.particles[:, 3] = (self.particles[:, 3] + 180.0) % 360.0 - 180.0

        particle_xz = self.particles[:, [0, 2]]
        reachable_xz = self.reachable[:, [0, 2]]
        diff = particle_xz[:, None, :] - reachable_xz[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(self.num_particles), nearest_idx]
        out_of_bounds = nearest_dist > dist_thred
        if np.any(out_of_bounds):
            # Snap implausible particles back onto the reachable manifold to prevent drift
            self.particles[out_of_bounds, :3] = self.reachable[nearest_idx[out_of_bounds]]

        # clip boundary
        # self.particles[:, 0] = self.particles[:, 0].clip(self.min_x, self.max_x)
        # self.particles[:, 2] = self.particles[:, 2].clip(self.min_z, self.max_z)


    def update(self, observation_features: np.ndarray, 
               confidence: np.ndarray,
               obs_world_positions: list[list[np.ndarray]], 
               radius: float,
               exp_name: str,) -> None:
        """Update particle weights based on observation features.
        
        Args:
            observation_features: CLIP features from current observation (M, feature_dim)
            radius: search radius for querying scene graph
        """
        scores = np.zeros(self.num_particles, dtype=np.float32)
        for i in range(self.num_particles):
            # Get features of objects near this particle
            nearby_features, nearby_obj_ids = query_scene_graph_features(  # TODO: consider view FOV
                self.particles[i], self.scene_graph, radius, self.fov
            )
            if len(nearby_features) == 0 or len(observation_features) == 0:
                scores[i] = 0.0
            else:
                # Compute cosine similarity between observation and scene graph features
                # observation_features: (M, D), nearby_features: (N, D)
                # Compute similarity matrix (M, N)
                visual_sim = observation_features @ nearby_features.T  # already normalized
 

                # Simple matching without reselection, would be better to use Hungarian algorithm
                match_sims = []
                match_pairs = []  # matched pair of current detected instances and \
                # the nearby objects of scene graph of [obs_idx, obj_idx]
                nearby_obj_positions = np.array([self.processed_scene_graph[obj_indice]['pose']
                                              for obj_indice in nearby_obj_ids])[:, [0, 2]]
                observation_positions = obs_world_positions[i][:, [0, 2]]
                distance_matrix = np.linalg.norm(
                    nearby_obj_positions[None, :, :] - observation_positions[:, None, :], axis=2
                )  # (M, N)
                sim_matrix = visual_sim
                for col in range(sim_matrix.shape[1]): 
                    # get the matched nearby instance for each detected object
                    sort_indices = np.argsort(sim_matrix[:, col])[::-1]
                    match_ins_idx = None
                    for idx in sort_indices:
                        if  sim_matrix[idx, col] > 0.25 and distance_matrix[idx, col] <= radius:
                            match_ins_idx = idx
                            break
                    if match_ins_idx is None:   
                        continue
                    if match_ins_idx in [row for (row, _) in match_pairs]:
                        # skip if this detection already explained another nearby object
                        continue
                    
                    if self.use_conf:
                        match_sims.append(sim_matrix[match_ins_idx, col] * confidence[match_ins_idx])
                    else:  # keep visual similarity only
                        match_sims.append(sim_matrix[match_ins_idx, col])
                    match_pairs.append([match_ins_idx, col])
                    sim_matrix[match_ins_idx, :] = -1.0  # prevent re-selection
                
                if len(match_pairs) == 0:
                    scores[i] = 0.0
                    continue
                
                match_sims = np.array(match_sims) 
                match_pairs = np.array(match_pairs)
                match_ins_idx = match_pairs[:, 0]  # detected from online observations
                match_obj_indices = nearby_obj_ids[match_pairs[:, 1]]  # queried from scene graph

                ## spatial similarity: time consuming to compute 3D IoU for each particle
                # spatial_sim = compute_3d_giou_accurate_batch(stack_pcd_bboxes(instance_pcd_particles[i]), 
                #                                stack_pcd_bboxes(self.get_pcd_bboxes(nearby_indices)))

                # match_spatial_sim = spatial_sim[match_pairs[:, 0], 
                #                                 match_pairs[:, 1]].cpu().numpy()


                ''' information gain
                landmarks that are semantically distinct, or are the centroid of cluster.
                '''
                if exp_name.startswith('baseline'):
                    scores[i] = np.mean(match_sims)
                else:
                    info_weights = np.array([self.processed_scene_graph[obj_idx]['info_weight'] 
                                            for obj_idx in match_obj_indices], dtype=np.float32)
                    scores[i] = np.mean(match_sims * info_weights)

        if len(observation_features) == 0:
            print('no observation features')
            # scores = np.ones_like(scores)  # if no observation, keep weights unchanged
            # likelihoods = 0.9 # decay with 0.9
            return
        # else:
        # Tempered softmax keeps numerical stability when similarity magnitudes vary per scene
        scaled = scores / self.temperature
        likelihoods = softmax(scaled)
        # print('likelihoods', likelihoods, 'scores', scores)
        self.weights *= likelihoods
        self.weights /= np.sum(self.weights)

    def resample(self) -> None:
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        samples = self.rng.random(self.num_particles)
        indices = np.searchsorted(cumulative, samples)
        self.particles = self.particles[indices]
        self.weights = np.ones_like(self.weights) / self.num_particles

    def estimate(self) -> Dict[str, float]:
        weighted_state = np.average(self.particles, axis=0, weights=self.weights)
        return {
            "x": float(weighted_state[0]),
            "y": float(weighted_state[1]),
            "z": float(weighted_state[2]),
            "rotation_y": float(weighted_state[3]),
        }

    def save_motion_deltas(self) -> None:
        if self.motion_delta_manager:
            self.motion_delta_manager.save()

    def _compute_obj_labels(self) -> List[str]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        tokenized_text = clip_tokenizer(self.class_set).to(device)
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

        # Classify each object embedding once so later weight lookups avoid GPU work
        labels: List[str] = []
        valid_index_set = set(self.obj_valid_indices.tolist())
        for idx, obj in enumerate(self.scene_graph):
            if idx not in valid_index_set:
                continue
            clip_ft = obj.get('clip_ft')
            if clip_ft is None:
                raise ValueError(f"Scene graph object {idx} is missing 'clip_ft' features.")
            with torch.no_grad():
                clip_tensor = clip_ft.to(device).unsqueeze(0)
                similarity = clip_tensor @ text_feat.T
                class_id = torch.argmax(similarity, dim=1).item()
            labels.append(self.class_set[class_id])
        return labels

    def _load_obj_labels_from_cache(self) -> List[str] | None:
        if not self.label_cache_path or not os.path.exists(self.label_cache_path):
            return None
        try:
            with open(self.label_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Failed to load object labels from {self.label_cache_path}: {exc}")
            return None

        labels = data.get("labels")
        cached_indices = data.get("valid_indices")
        cached_class_set = data.get("class_set")
        expected_indices = self.obj_valid_indices.tolist()
        if (
            cached_indices != expected_indices
            or not isinstance(labels, list)
            or len(labels) != len(expected_indices)
        ):
            print(f"Label cache {self.label_cache_path} is incompatible with current scene graph; recomputing labels.")
            return None
        if cached_class_set is not None and sorted(list(cached_class_set)) != sorted(self.class_set):
            print(f"Label cache {self.label_cache_path} was generated with different classes; recomputing labels.")
            return None
        return [str(label) for label in labels]

    def _save_obj_labels_cache(self) -> None:
        if not self.label_cache_path:
            return
        payload = {
            "labels": self.obj_labels,
            "valid_indices": self.obj_valid_indices.tolist(),
            "class_set": self.class_set,
        }
        save_dir = os.path.dirname(self.label_cache_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        try:
            with open(self.label_cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved object labels to {self.label_cache_path}")
        except OSError as exc:
            print(f"Warning: failed to save label cache to {self.label_cache_path}: {exc}")

