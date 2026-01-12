"""Semantic localization demo that reuses trajectories sampled from reachable
positions and maintains a particle filter against a 3D scene graph.

The script is intentionally lightweight: it borrows conventions from
`generate_ai2thor_dataset.py`, interacts with AI2-THOR, and tracks a
probabilistic pose estimate while the agent follows a random walk.
"""
from __future__ import annotations

import argparse
import json
import os

if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    # CuBLAS needs a workspace hint for deterministic kernels on CUDA >= 10.2.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import cv2
import numpy as np
import open3d as o3d
import open_clip
import torch
import torchvision
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import distinctipy
import gzip
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from conceptgraph.utils.ai2thor import (
    compute_pose,
    get_agent_pose_from_event,
    get_scene,
)
from conceptgraph.slam.slam_classes import MapObjectList
from semantic_particle_filter import SemanticParticleFilter
from semantic_detection import SemanticDetectors
from localization_utils import (softmax, 
                      semantic_similarity, 
                      Pose, 
                      extract_visible_semantics, 
                      load_reachable_positions,
                      load_scene_graph,
                    #   sample_pose_uniform,
                      process_ai2thor_classes,
                      compute_clip_features,
                      compute_intrinsics,
                      visualize_localization_step,
                      vis_trajectory,
                      ensure_output_dir
)
from conceptgraph.slam.utils import create_object_pcd, process_pcd, get_bounding_box



def set_global_seeds(seed: int) -> None:
    """Seed all major randomness sources and force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
class ObservationCache:
    """Record or replay detector outputs for deterministic measurement updates."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.entries: List[Dict[str, np.ndarray]] = []
        self.cursor = 0
        self.replay_entries: List[Dict[str, np.ndarray]] | None = None
        if path and os.path.exists(path):
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
            self.replay_entries = data.get("entries", [])
            print(f"Loaded observation cache from {path} ({len(self.replay_entries)} steps)")

    def next(self) -> Dict[str, np.ndarray] | None:
        if self.replay_entries is None:
            return None
        if self.cursor >= len(self.replay_entries):
            raise ValueError(
                f"Observation cache {self.path} only stores {len(self.replay_entries)} steps, "
                "but this run needs more."
            )
        entry = self.replay_entries[self.cursor]
        self.cursor += 1
        return entry

    def record(self, features: np.ndarray, masks: np.ndarray, confidence: np.ndarray) -> None:
        if self.path is None or self.replay_entries is not None:
            return
        feature_array = np.array(features, dtype=np.float32, copy=True)
        mask_array = np.array(masks, dtype=bool, copy=True)
        confidence_array = np.array(confidence, dtype=np.float32, copy=True)
        self.entries.append({"features": feature_array, "masks": mask_array, "confidence": confidence_array})

    def save(self) -> None:
        if (
            self.path is None
            or self.replay_entries is not None
            or len(self.entries) == 0
        ):
            return
        save_dir = os.path.dirname(self.path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with gzip.open(self.path, "wb") as f:
            pickle.dump({"entries": self.entries}, f)
        print(f"Saved observation cache to {self.path}")


def load_result(result_path):
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])

        '''
        ['image_idx', 'mask_idx', 'color_path', 'class_name', 'class_id', 'num_detections', 
        'mask', 'xyxy', 'conf', 'n_points', 'pixel_area', 'contain_number', 'inst_color', 
        'is_background', 'clip_ft', 'text_ft', 'pcd', 'bbox']

        bbox: o3d.geometry.OrientedBoundingBox
        clip_ft: 1024
        text_ft: 1024
        '''
        print(f"Loaded bjects", objects[0]['clip_ft'].shape, objects[0]['text_ft'].shape)
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])
            print(f"Loaded bg_objects", bg_objects[0])
        class_colors = results['class_colors']
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)

        bg_objects = None
        class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
        class_colors = {str(i): c for i, c in enumerate(class_colors)}
    else:
        raise ValueError("Unknown results type: ", type(results))

    return objects, bg_objects, class_colors


def sample_random_trajectory(
    reachable: Sequence[Dict[str, float]],
    num_steps: int,
    step_radius: float,
    trajectory_file: str='',
    is_random: bool = False,
) -> List[Pose]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    
    if not is_random and os.path.exists(trajectory_file) and trajectory_file.endswith('.pkl'):
        print(f"Loading trajectory from {trajectory_file}")
        with open(trajectory_file, 'rb') as f:
            traj: List[dict] = pickle.load(f)
        poses: List[Pose] = []
        for pose_dict in traj:
            poses.append(
                Pose(
                    position=pose_dict['position'],
                    rotation=pose_dict['rotation'],
                    horizon=pose_dict.get('horizon', 0.0)
                )
            )
        return poses
    else:
        print(f"Sampling new trajectory and saving to {trajectory_file}")
        traj: List[dict] = []
        poses: List[Pose] = []
        current = random.choice(reachable)
        heading = random.choice([0.0, 90.0, 180.0, -90.0])
        # [0, 45, 90, 135, 180, 225, 270, 315]
        poses.append(
            Pose(position=current, rotation={"x": 0.0, "y": heading, "z": 0.0}, horizon=0.0)
        )
        traj.append({
                'position': current,
                'rotation': {"x": 0.0, "y": heading, "z": 0.0},
            })
        pos_key = lambda p: (p["x"], p["y"], p["z"])
        visited_positions = set(pos_key(current))
        # In the camera coordinate, Z is the viewing direction, X is right, and Y is up. 
        for i in range(1, num_steps):
            candidates = [
                p for p in reachable 
                if np.linalg.norm((np.array(list(p.values())) - 
                                  np.array(list(current.values())))) <= step_radius
            ]
            # prioritize unvisited candidates
            # unvisited_candidates = [
            #     p for p in candidates if pos_key(p) not in visited_positions
            # ]
            # candidate_pool = unvisited_candidates if unvisited_candidates else candidates
            # current = random.choice(candidate_pool if candidate_pool else reachable)
            # visited_positions.add(pos_key(current))

            current = random.choice(candidates if candidates else reachable)
            heading += random.choice([-90.0, 0.0, 90.0])
            poses.append(
                Pose(position=current, rotation={"x": 0.0, "y": heading, "z": 0.0}, horizon=0.0)
            )
            traj.append({
                'position': current,
                'rotation': {"x": 0.0, "y": heading, "z": 0.0},
            })


        # Save trajectory
        with open(trajectory_file, 'wb') as f:
            pickle.dump(traj, f)
        return poses


def compute_control(prev_pose: Pose, next_pose: Pose) -> tuple[np.ndarray, float]:
    prev_vec = prev_pose.as_vector()
    next_vec = next_pose.as_vector()
    delta_translation = next_vec[:3] - prev_vec[:3]
    delta_rotation = next_vec[3] - prev_vec[3]
    delta_rotation = (delta_rotation + 180.0) % 360.0 - 180.0
    return delta_translation, delta_rotation



def all_instances_pcd_projection(seg_masks, depth_map, intrinsics, particle_pose_world):
    """
    Project instance masks into 3D using the depth map (camera frame) and then into
    the world frame hypothesized by each particle.

    Args:
        seg_masks (np.ndarray): (N_inst, H, W) boolean masks.
        depth (np.ndarray): (H, W) depth map in meters.
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        particle_pose_world (np.ndarray): (N_particles, 4) pose vectors [x, y, z, yaw_deg]
                                          or (N_particles, 4, 4) transform matrices.

    Returns:
        list[list[np.ndarray]]: world coordinates per particle per instance mask.
    """

    def pose_vec_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
        if pose_vec.shape == (4, 4):
            return pose_vec
        if pose_vec.shape[-1] != 4:
            raise ValueError(f"Unexpected particle pose shape {pose_vec.shape}")
        x, y, z, yaw = pose_vec
        yaw_rad = np.deg2rad(yaw)
        cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        transform = np.eye(4, dtype=np.float32)
        transform[0, 0] = cos_yaw
        transform[0, 2] = sin_yaw
        transform[2, 0] = -sin_yaw
        transform[2, 2] = cos_yaw
        transform[:3, 3] = [x, y, z]
        return transform

    if seg_masks is None or len(seg_masks) == 0:
        return [[] for _ in range(len(particle_pose_world))]

    inv_intrinsics = np.linalg.inv(intrinsics).astype(np.float32)  # 3x3 K matrix
    instance_3D_world_particles: list[list[np.ndarray]] = []

    for particle_pose in particle_pose_world:
        world_T_camera = pose_vec_to_matrix(np.asarray(particle_pose, dtype=np.float32))  # [R, t]
        instance_world_points: list[np.ndarray] = []

        for mask in seg_masks:
            ys, xs = np.where(mask)
            depth = depth_map[ys, xs]

            pixels = np.stack(
                (xs.astype(np.float32),
                 ys.astype(np.float32),
                 np.ones_like(xs, dtype=np.float32)),
                axis=0,
            )  # [u, v, 1].T
            camera_dirs = inv_intrinsics @ pixels
            camera_points = camera_dirs * depth[np.newaxis, :]


            # [x, y, z, 1].T x N
            homog_points = np.vstack(
                (camera_points, np.ones((1, camera_points.shape[1]), dtype=np.float32))
            ) 

            # x right, y up, z forward
            world_points = (world_T_camera @ homog_points)[:3].T  # (N, 4)

            centroid = np.mean(world_points, axis=0)
            instance_world_points.append(centroid)

            # Create an Open3D PointCloud object
            ## Perturb the points a bit to avoid colinearity
            # world_points += np.random.normal(0, 4e-3, world_points.shape)
            # if world_points.shape[0] == 0:
            #     import pdb; pdb.set_trace()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(world_points)
            # # get largest cluster, filter out noise 
            # # pcd = process_pcd(pcd, None)
            
            # pcd_bbox = get_bounding_box(None, pcd)
            # # pcd_bbox.color = [0,1,0]
            
            # if pcd_bbox.volume() < 1e-6:
            #     instance_world_points.append(np.empty((0, 3), dtype=np.float32))
            #     print("Warning: bbox volume too small")
            #     continue
            # print(pcd_bbox)
            # instance_world_points.append(pcd_bbox)

        assert len(instance_world_points) == len(seg_masks)
        instance_3D_world_particles.append(np.array(instance_world_points))
    assert len(instance_3D_world_particles) == len(particle_pose_world)
    return instance_3D_world_particles

    

def get_instance_pcd(depth_array,
                     mask,
                     cam_K,
                    #  image,
                     global_pose,
                     ):
    camera_object_pcd = create_object_pcd(
        depth_array,
        mask,
        cam_K,
        # image,
        obj_color = None
    )
    
    # It at least contains 5 points
    if len(camera_object_pcd.points) < 5: 
        return None
    
    if global_pose is not None:
        global_object_pcd = camera_object_pcd.transform(global_pose)
    else:
        global_object_pcd = camera_object_pcd
    
    # get largest cluster, filter out noise 
    global_object_pcd = process_pcd(global_object_pcd)
    
    pcd_bbox = get_bounding_box(global_object_pcd)
    # pcd_bbox.color = [0,1,0]
    
    if pcd_bbox.volume() < 1e-6:
        return None
    
    return global_object_pcd, pcd_bbox


def follow_trajectory(
    controller: Controller,
    trajectory: Sequence[Pose],
    particle_filter: SemanticParticleFilter,
    semantic_detectors: SemanticDetectors,
    reachable_positions: List,
    scene_graph_attributes: Dict[str, Any],
    intrinsics: np.ndarray,
    measurement_radius: float,
    exp_name: str,
    use_confidence: bool,
    observation_cache: ObservationCache | None = None,
) -> Dict[str, Any]:
    log: Dict[str, Any] = {
        "agent_poses": [],
        "estimated_poses": [],
        "visible_semantics": [],
        "obs_features": [],
    }

    # vis_trajectory(trajectory)
    
    # List to store visualization frames for gif
    frames = []
    xy_errors = []
    yaw_errors = []
    for i in trange(len(trajectory), desc="localizing"):
        pose = trajectory[i]
        # print(f"Step {i}: moving to pose {pose}")
        # limit the horizon to [-30, 60]
        # horizon = pose.horizon
        # horizon = max(min(horizon, 60-1e-6), -30+1e-6)

        event = controller.step(
            action="Teleport",
            position=pose.position,
            rotation=pose.rotation,
            horizon=pose.horizon,
            standing=True,
            forceAction=True,
        )
        if not event.metadata["lastActionSuccess"]:
            # raise Exception(event.metadata["errorMessage"])

            # Seems that the teleportation failures are based on position. 
            # Once it fails on a position, it will fail on all orientations.
            # Therefore, we can simply skip these failed trials. 
            print("Failed to teleport to the position.", pose["position"], pose["rotation"])
            continue
        
        # get observation
        color = np.asarray(event.frame).copy()
        depth_map = np.asarray(event.depth_frame).copy()
        depth_map[depth_map > 15] = 0  # Cut off the depth at 15 meters 
        instance = np.asarray(event.instance_segmentation_frame).copy()


        agent_pose = get_agent_pose_from_event(event)
        # camera_pose = get_camera_pose_from_event(event)
        # print('agent_pose', agent_pose) #, 'camera_pose', camera_pose)

        # Extract both semantics (for logging) and features (for localization)
        gt_semantics = extract_visible_semantics(event)  # filtering Floor, Ceiling, Wall
        print('extract semantics from observation via simulator:', gt_semantics)
        
        
        # TODO: Extract CLIP features from event - this needs to be implemented
        
        cached_obs = observation_cache.next() if observation_cache else None
        if cached_obs is None:
            image_crops, image_feats, text_feats, seg_masks, confidence = \
                semantic_detectors.detect(color)
            obs_features = image_feats
            if observation_cache:
                observation_cache.record(obs_features, seg_masks, confidence)
        else:
            obs_features = cached_obs["features"]
            seg_masks = cached_obs["masks"]
            if use_confidence:
                confidence = cached_obs["confidence"]
            else:
                confidence = None

        if i > 0:
            delta_t, delta_r = compute_control(trajectory[i - 1], pose)
            particle_filter.predict(delta_t, delta_r)
        
        valid_mask_indices = []
        for idx, mask in enumerate(seg_masks):
            ys, xs = np.where(mask)
            if xs.size == 0:
                print("Warning: empty mask")
                continue

            depths = depth_map[ys, xs]
            valid = depths > 0
            if not np.any(valid):
                print("Warning: no valid depth")
                continue
            valid_mask_indices.append(idx)
        
        seg_masks = seg_masks[valid_mask_indices]
        obs_features = obs_features[valid_mask_indices]
        if confidence is not None:
            confidence = confidence[valid_mask_indices]

        obs_world_positions = \
            all_instances_pcd_projection(seg_masks, depth_map, intrinsics, particle_filter.particles)


        particle_filter.update(obs_features, confidence, obs_world_positions, 
                               measurement_radius, exp_name)
        particle_filter.resample()
        estimated_pose = particle_filter.estimate()
        # print('estimated_pose', estimated_pose)  # 'agent_pose', agent_pose, 
        log["agent_poses"].append(agent_pose.tolist())
        log["estimated_poses"].append(estimated_pose)
        log["visible_semantics"].append(gt_semantics)
        log["obs_features"].append(obs_features.tolist())
        
        
        # compute metrics
        agent_rotation_y = np.arctan2(agent_pose[0, 2], agent_pose[2, 2]) * 180.0 / np.pi
        agent_pose_vec = [agent_pose[0, 3], agent_pose[1, 3], agent_pose[2, 3], agent_rotation_y]
        agent_pose_vec = np.array(agent_pose_vec)

        estimated_pose_vec = [estimated_pose['x'], estimated_pose['y'], estimated_pose['z'], 
                              estimated_pose['rotation_y']]
        estimated_pose_vec = np.array(estimated_pose_vec)
        xy_error = np.linalg.norm((agent_pose_vec - estimated_pose_vec)[[0, 2]]) # neglect height
        yaw_error = np.abs(estimated_pose_vec[3] - agent_pose_vec[3])
        if yaw_error > 180: yaw_error = 360 - yaw_error
        xy_errors.append(xy_error)
        yaw_errors.append(yaw_error)
        overall_error = xy_error + np.radians(yaw_error)  # simple sum
        # print(f"Step {i}: Overall Error = {overall_error:.3f}")
        print(f"Step {i}: XY Error = {xy_error:.3f} m, Yaw Error = {yaw_error:.3f} degrees")

        # Visualize current step
        frame = visualize_localization_step(
            agent_pose=agent_pose,
            estimated_pose=estimated_pose,
            particles=particle_filter.particles,
            weights=particle_filter.weights,
            step=i,
            reachable_positions=reachable_positions,
            scene_graph_attributes=scene_graph_attributes,
            trajectory=trajectory,
        )
        frames.append(frame)
        # print('gt pose:', get_agent_pose_from_event(event), 'estimated pose:', log["estimated_poses"][-1])

        # log["observation_features"].append(observation_features.tolist() if len(observation_features) > 0 else [])
    
    # Store frames in log
    log["frames"] = frames
        
    # write errors to log
    log['xy_errors'] = xy_errors
    log['yaw_errors'] = yaw_errors
    return log


def build_controller(args: argparse.Namespace) -> Controller:
    return Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene=get_scene(args.scene_name),
        gridSize=args.grid_size,
        snapToGrid=False,
        rotateStepDegrees=90,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        renderSemanticSegmentation=True,
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
        platform=CloudRendering,
    )



def run(args: argparse.Namespace) -> None:
    set_global_seeds(args.seed)
    output_dir = ensure_output_dir(args)
    reachable_path = args.data_root + '/' + args.scene_name + '/' + args.reachable_file
    scene_graph_path = args.data_root + '/' + args.scene_name + '/pcd_saves/' + args.scene_graph_file
    
    controller = build_controller(args)
    
    if os.path.exists(reachable_path):
        reachable_positions = load_reachable_positions(reachable_path)
    else:
        reachable_positions = controller.step(
            action="GetReachablePositions", raise_for_failure=True
        ).metadata["actionReturn"]

    scene_graph, bg_objects, class_colors = load_scene_graph(scene_graph_path)

    # load all the object classes in the scene
    obj_meta_path = args.data_root + '/' + args.scene_name + '/' + "obj_meta.json"
    with open(obj_meta_path, "r") as f:
        obj_meta = json.load(f)
    
    # Get a list of object classes in the scene
    classes = process_ai2thor_classes(
        [obj["objectType"] for obj in obj_meta],
        add_classes=[],
        remove_classes=['wall', 'floor', 'room', 'ceiling']
    )

    intrinsics = compute_intrinsics(args.fov, args.width, args.height)

    '''
    scene_graph: ['class_name', 'class_id', 'is_background', 'clip_ft', 'text_ft', 'bbox']
    '''
    semantic_detectors = SemanticDetectors(args, classes)

    
    trajectory = sample_random_trajectory(
        reachable_positions,
        num_steps=args.trajectory_length,
        step_radius=args.step_radius,
        trajectory_file=output_dir + '/' + args.trajectory_file,
        is_random=args.random,
    )


    init_particles_path = (
        os.path.join(output_dir, args.init_particles_file)
        if args.init_particles_file
        else None
    )
    motion_delta_path = (
        os.path.join(output_dir, args.motion_delta_file)
        if args.motion_delta_file
        else None
    )
    obj_labels_path = (
        os.path.join(output_dir, args.obj_labels_file)
        if args.obj_labels_file
        else None
    )
    observation_cache_path = (
        os.path.join(output_dir, args.observation_cache_file)
        if args.observation_cache_file
        else None
    )
    observation_cache = ObservationCache(observation_cache_path) if observation_cache_path else None
    rng = np.random.default_rng(args.seed)

    # initialize particle filter
    particle_filter = SemanticParticleFilter(
        reachable_positions,
        scene_graph,
        class_set=classes,
        num_particles=args.num_particles,
        motion_noise=args.motion_noise,
        rotation_noise=args.rotation_noise,
        random_state=rng,
        temperature=args.obs_temperature,
        fov=args.fov,
        alpha=args.alpha,
        use_conf=args.use_conf,
        init_particles_path=init_particles_path,
        motion_delta_path=motion_delta_path,
        label_cache_path=obj_labels_path,
    )
    scene_graph_attributes = {
        'positions': particle_filter.obj_positions,
        'visible_areas': particle_filter.visible_areas,
        'cluster_labels': particle_filter.cluster_labels,
    }


    #===============main process=================
    log = follow_trajectory(
        controller,
        trajectory,
        particle_filter,
        semantic_detectors,
        reachable_positions,
        scene_graph_attributes,  # used for visualization only
        intrinsics,
        args.measurement_radius,
        args.exp_name,
        args.use_conf,
        observation_cache,
    )
    #============================================


    if observation_cache:
        observation_cache.save()
    # particle_filter.save_motion_deltas() 


    # Save log
    args.device = str(args.device)
    exp_name = '_' + args.exp_name if args.exp_name != '' else ''
    log_path = output_dir + '/' + f"localization_log{exp_name}.json"
    summary = {
        "scene_name": args.scene_name,
        "trajectory": [pose.as_vector().tolist() for pose in trajectory],
        "agent_poses": log["agent_poses"],
        "estimated_poses": log["estimated_poses"],
        "visible_semantics": log["visible_semantics"],
        'xy_errors': log['xy_errors'],
        'yaw_errors': log['yaw_errors'],
        "config": vars(args),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved localization log to {log_path}")
    
    # Save visualization as mp4
    if "frames" in log and log["frames"]:
        video_path = os.path.join(output_dir, f"localization_visualization{exp_name}.mp4")
        imageio.mimwrite(video_path, log["frames"], fps=2, quality=8)  # type: ignore[arg-type]
        print(f"Saved visualization video to {video_path}")

    # Plot errors over time
    xy_errors = log['xy_errors']
    yaw_errors = log['yaw_errors']
    x_bar = np.arange(len(xy_errors))
    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(x_bar, xy_errors, label='XY Error (m)')
    ax[0].scatter(x_bar, xy_errors)
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('XY Error (m)')
    ax[0].set_title('Localization XY Error over Time')
    ax[0].set_yticks(np.arange(0, int(max(xy_errors)) + 0.5, 0.5))
    ax[0].legend()

    ax[1].plot(x_bar, yaw_errors, label='Orientation Error (m)')
    ax[1].scatter(x_bar, yaw_errors)
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Orientation Error (degrees)')
    ax[1].set_title('Localization Orientation Error over Time')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'localization_errors{exp_name}.png'))
    plt.show()

    
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-vocabulary localization demo")
    parser.add_argument("--scene_name", default="train_3_uniform")
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--grid_size", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=int, default=90)
    
    parser.add_argument("--trajectory_file", type=str, default='trajectory2.pkl')
    parser.add_argument(
        "--init_particles_file",
        type=str,
        default='init_particles.npy',
        help="File name or path used to cache the initial particle set (.npy).",
    )
    parser.add_argument(
        "--motion_delta_file",
        type=str,
        default='motion_deltas.pkl',
        help="File used to record/replay sampled motion deltas (.pkl).",
    )
    parser.add_argument(
        "--obj_labels_file",
        type=str,
        default='obj_labels.json',
        help="File used to cache CLIP-based object labels (.json). Use '' to recompute each run.",
    )
    parser.add_argument(
        "--observation_cache_file",
        type=str,
        default='observation_cache.pkl',
        help="File used to record/replay detector outputs (.pkl). Use '' to disable.",
    )
    parser.add_argument("--trajectory_length", type=int, default=40)
    parser.add_argument("--intrinsic_file", type=str, default='intrinsic.pkl')
    parser.add_argument("--alpha", type=float, default=1.0) # 
    parser.add_argument("--use_conf", action="store_true") # 
    parser.add_argument("--step_radius", type=float, default=2.0) # 
    parser.add_argument("--num_particles", type=int, default=256)
    parser.add_argument("--motion_noise", type=float, default=0.05)
    parser.add_argument("--rotation_noise", type=float, default=5.0)
    parser.add_argument("--measurement_radius", type=float, default=4)  # 4
    parser.add_argument("--obs_temperature", type=float, default=1.0,
                        help="Softmax temperature for the semantic observation model.")
    parser.add_argument("--data_root", type=str, default=str(Path("~/ldata/ai2thor").expanduser()))
    parser.add_argument("--reachable_file", type=str, default='reachable.json')
    parser.add_argument("--scene_graph_file", type=str, default='full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz')
    parser.add_argument(
        "--output_root",
        type=str,
        default="loc_outputs",
    )
    parser.add_argument("--sam_variant", type=str, default="sam",
                        choices=['fastsam', 'mobilesam', "lighthqsam"])
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.3)
    parser.add_argument("--nms_threshold", type=float, default=0.6)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--random", action="store_true", help="generate a new random trajectory for localization.")
    parser.add_argument("--device", type=torch.device, 
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
