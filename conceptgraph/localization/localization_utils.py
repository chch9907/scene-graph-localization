import argparse
import copy
from dataclasses import dataclass
import gzip
import os
from pathlib import Path
import pickle
import random
import re
from typing import Any, Dict, List, Sequence
import distinctipy
from matplotlib import pyplot as plt
import torch
from PIL import Image
import open3d as o3d
import numpy as np
import json

from conceptgraph.slam.slam_classes import MapObjectList
from ai2thor.controller import Controller
from matplotlib.patches import Circle

def softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values)
    if denom <= 0.0:
        return np.ones_like(values) / len(values)
    return exp_values / denom

def semantic_similarity(observed: set[str], expected: set[str]) -> float:
    if not observed and not expected:
        return 1.0
    if not observed or not expected:
        return 0.1
    intersection = len(observed & expected)
    union = len(observed | expected)
    return (intersection / union) if union > 0 else 0.1


def compute_match_batch(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Compute object association based on spatial and visual similarities
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of binary values, indicating whether a detection is associate with an object. 
        Each row has at most one 1, indicating one detection can be associated with at most one existing object.
        One existing object can receive multiple new detections
    '''
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        # Fuse geometric and visual signals before committing to an association
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
        row_max, row_argmax = torch.max(sims, dim=1) # (M,), (M,)
        for i in row_max.argsort(descending=True):
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return assign_mat


@dataclass
class Pose:
    position: Dict[str, float]
    rotation: Dict[str, float]
    horizon: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.position["x"],
                self.position["y"],
                self.position["z"],
                self.rotation["y"],
                self.horizon,
            ],
            dtype=np.float32,
        )

# def load_scene_graph(path: Path) -> Dict[str, Any]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

def extract_visible_semantics(event: Any) -> List[str]:
    semantics: List[str] = []
    for obj in event.metadata.get("objects", []):
        if obj.get("visible") and not obj.get("isPickedUp"):
            obj_class = obj.get("objectType", obj.get("name", "unknown"))
            if obj_class not in ["Floor", "Ceiling", "Wall"]:
                semantics.append(obj_class)
    return semantics


def load_reachable_positions(result_path: Path) -> List[Dict[str, float]]:
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("reachable.json must contain a list of positions")
    return [
        {"x": float(pos["x"]), "y": float(pos.get("y", 0.0)), "z": float(pos["z"])}
        for pos in data
    ]

def load_scene_graph(result_path):
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
        # print(f"Loaded bjects", objects[0]['clip_ft'].shape, objects[0]['text_ft'].shape)
        # print([obj['clip_ft'] for obj in objects])
        # assert False
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])
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


def sample_pose_uniform(controller: Controller, n_positions: int=-1):
    """
    Uniformly sample n_positions from the reachable positions
    for each position, uniformly sample 8 rotations (0, 45, 90, 135, 180, 225, 270, 315)
    """
    reachable_positions = controller.step(action="GetReachablePositions").metadata[
        "actionReturn"
    ]

    # Convert the positions to numpy array
    reachable_np = np.array([[p["x"], p["y"], p["z"]] for p in reachable_positions])
    # Sort the positions by z, x
    sort_idx = np.lexsort((reachable_np[:, 2], reachable_np[:, 0]))
    reachable_positions = [reachable_positions[i] for i in sort_idx]

    # Randomly sample n_positions. This is a temporal hack for uniform sampling.
    if n_positions < 0:
        n_positions = len(reachable_positions)
    else:
        n_positions = min(n_positions, len(reachable_positions))

    sampled_positions = np.random.choice(
        reachable_positions, n_positions, replace=False
    )

    # Generate a list of poses for taking pictures
    sampled_poses = []
    # sampled_poses: List[Pose] = []
    for position in sampled_positions:

        heading = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        sampled_poses.append(
            Pose(position=position, 
                rotation={"x": 0.0, "y": heading, "z": 0.0}, 
                horizon=0.0,
                )
        )
        # for rot_y in [0, 45, 90, 135, 180, 225, 270, 315]:
        #     rotation = dict(x=0, y=rot_y, z=0)

        #     sampled_poses.append(
        #         dict(
        #             position=position,
        #             rotation=rotation,
        #             horizon=0,
        #             standing=True,
        #         )
        #     )

    return sampled_poses

def process_ai2thor_classes(classes: List[str], add_classes:List[str]=[], remove_classes:List[str]=[]) -> List[str]:
    '''
    Some pre-processing on AI2Thor objectTypes in a scene
    '''

    classes = list(set(classes))
    

    for c in add_classes:
        classes.append(c)
        
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    # Split the element in classes by captical letters
    classes = [obj_class.replace("TV", "Tv") for obj_class in classes]
    classes = [re.findall('[A-Z][^A-Z]*', obj_class) for obj_class in classes]

    # Join the elements in classes by space
    classes = [" ".join(obj_class) for obj_class in classes]
    
    return classes

def compute_clip_features(pil_image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    backup_image = pil_image.copy()
    
    # image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_feats = []

    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = pil_image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
        
        # Get the preprocessed image for clip from the crop 
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats

def compute_intrinsics(vfov, height, width):
    """
    Compute the camera intrinsics matrix K from the
    vertical field of view (in degree), height, and width.
    """
    # For Unity, the field view is the vertical field of view.
    f = height / (2 * np.tan(np.deg2rad(vfov) / 2))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

# def get_new_pose(pose, rel_pose_change):
#     """taken from chaplot's ANS: 
#     pose is in world coordinate.
#     rel_pose_change is in local coordinate.
#     """
#     if len(pose.shape) > 1:
#         x, y, o = pose[:, 0], pose[:, 1], pose[:, 2]
#         dx, dy, do = rel_pose_change[:, 0], rel_pose_change[:, 1], rel_pose_change[:, 2]
#     else:
#         x, y, o = pose
#         dx, dy, do = rel_pose_change

#     global_dx = dx * np.sin(o) + dy * np.cos(o)
#     global_dy = dx * np.cos(o) - dy * np.sin(o)
#     x += global_dy
#     y += global_dx
#     o += do

#     if len(pose.shape) > 1:
#         for i in range(len(o)):
#             o[i] = _normalize_heading(o[i])
#         return np.stack([x, y, o], axis=1)
#     else:
#         o = _normalize_heading(o)
#         return np.array([x, y, o])

# def pixel_to_world(depth_intrinsics, center_2d, distance):
#     '''coordinate definition:
#     https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0?fbclid=IwAR3gogVZe824YUps88Dzp02AN_XzEm1BDb0UbmzfoYvn1qDFb7KzbIz9twU#pixel-coordinates
#     input: pixel coordinate
#     output: point coordinate
#     '''
#     point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, center_2d, distance)  # x, y, depth
#     # point[0], point[1], point[2] = right, down, forward
#     # transform to SLAM right-hand coordinate: x, y, z = point[2], -point[0], -point[1]
#     return [point[2], -point[0], -point[1]]


def vis_trajectory(trajectory: Sequence[Pose]) -> None:
    xs = [pose.position.get("x", 0.0) for pose in trajectory]
    zs = [pose.position.get("z", 0.0) for pose in trajectory]
    plt.figure(figsize=(4, 4))
    plt.plot(xs, zs, marker="o", linewidth=1, markersize=3)
    plt.scatter(xs[0], zs[0], c="green", label="start")
    plt.scatter(xs[-1], zs[-1], c="red", label="end")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Sampled 2D trajectory")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_localization_step(
    agent_pose: np.ndarray,
    estimated_pose: Dict[str, float],
    particles: np.ndarray,
    weights: np.ndarray,
    step: int,
    reachable_positions: List,
    scene_graph_attributes: Dict[str, Any],
    arrow_len: float = 0.6,
    trajectory: Sequence[Pose] = None,
) -> np.ndarray:
    """
    Visualize agent pose, estimated pose, and particles at a given step.
    
    Args:
        agent_pose: 4x4 transformation matrix of ground truth agent pose
        estimated_pose: dict with keys x, y, z, rotation_y
        particles: (N, 4) array of particles [x, y, z, rotation_y]
        weights: (N,) array of particle weights
        step: current step number
        trajectory: optional trajectory to show in background
    
    Returns:
        RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract agent position from 4x4 matrix
    agent_x = agent_pose[0, 3]
    agent_z = agent_pose[2, 3]
    # Extract rotation from rotation matrix
    agent_rotation_y = np.arctan2(agent_pose[0, 2], agent_pose[2, 2]) * 180.0 / np.pi
    

    # reachable areas
    xs = [rp["x"] for rp in reachable_positions]
    zs = [rp["z"] for rp in reachable_positions]

    min_x_axis, max_x_axis = min(xs), max(xs)
    min_z_axis, max_z_axis = min(zs), max(zs)
    ax.scatter(xs, zs,
               alpha=0.5,
               c='gray',
               label='reachable')
    
    # object positions
    obj_positions = scene_graph_attributes['positions']
    ax.scatter(obj_positions[:, 0], obj_positions[:, 2],
            #    alpha=0,
               c='orange',
               label='object positions')
    
    # visible areas
    # vis_xs = []
    # vis_zs = []
    # obj_visible_areas = scene_graph_attributes['visible_areas']
    # for area in obj_visible_areas:
    #     if len(area) > 0: # area is an array
    #         # print(area)
    #         vis_xs.extend(list(area[:, 0]))
    #         vis_zs.extend(list(area[:, 1]))
    # ax.scatter(vis_xs, vis_zs,
    #             alpha=0.5,
    #             c='green',
    #             label='visible areas')

    # cluster labels
    # obj_cluster_labels = scene_graph_attributes['cluster_labels']
    # unique_labels = np.unique(obj_cluster_labels)
    # for idx, label in enumerate(unique_labels):
    #     mask = obj_cluster_labels == label
    #     cluster_points = obj_positions[mask]
    #     if cluster_points.size == 0:
    #         continue
    #     xz_points = cluster_points[:, [0, 2]]
    #     center = xz_points.mean(axis=0)
    #     radii = np.linalg.norm(xz_points - center, axis=1)
    #     radius = float(np.percentile(radii, 90)) if radii.size else 0.2
    #     radius = max(radius, 0.2)
    #     circle = Circle(
    #         (center[0], center[1]),
    #         radius,
    #         fill=False,
    #         linestyle='--',
    #         linewidth=1.5,
    #         alpha=0.5,
    #         edgecolor=plt.cm.tab20(idx % 20),
    #         # label=f'Cluster {label}',
    #     )
    #     ax.add_patch(circle)

    


    # Plot trajectory if provided
    if trajectory is not None:
        xs = [pose.position.get("x", 0.0) for pose in trajectory]
        zs = [pose.position.get("z", 0.0) for pose in trajectory]
        ax.plot(xs, zs, 'k-', alpha=0.3, linewidth=1, label='Trajectory')
    
    # Plot particles (color by weight)
    particle_sizes = weights * 1000 + 10  # Scale weights to visible sizes
    scatter = ax.scatter(
        particles[:, 0],
        particles[:, 2],
        c=weights,
        s=particle_sizes,
        cmap='viridis',
        alpha=0.5,
        label='Particles'
    )
    
    # Plot estimated pose
    ax.scatter(
        estimated_pose["x"],
        estimated_pose["z"],
        c='blue',
        s=200,
        marker='*',
        edgecolors='darkblue',
        linewidths=2,
        label='Estimated Pose',
        zorder=5
    )
    # Draw heading arrow for estimated pose
    
    est_dx = arrow_len * np.sin(estimated_pose["rotation_y"] * np.pi / 180.0)
    est_dz = arrow_len * np.cos(estimated_pose["rotation_y"] * np.pi / 180.0)
    ax.arrow(
        estimated_pose["x"],
        estimated_pose["z"],
        est_dx,
        est_dz,
        head_width=0.15,
        head_length=0.1,
        fc='blue',
        ec='blue',
        linewidth=2,
        zorder=5
    )
    
    # Plot ground truth agent pose
    ax.scatter(
        agent_x,
        agent_z,
        c='red',
        s=200,
        marker='o',
        edgecolors='darkred',
        linewidths=2,
        label='Ground Truth',
        zorder=10
    )
    # Draw heading arrow for agent
    agent_dx = arrow_len * np.sin(agent_rotation_y * np.pi / 180.0)
    agent_dz = arrow_len * np.cos(agent_rotation_y * np.pi / 180.0)
    ax.arrow(
        agent_x,
        agent_z,
        agent_dx,
        agent_dz,
        head_width=0.15,
        head_length=0.1,
        fc='red',
        ec='red',
        linewidth=2,
        zorder=10
    )
    
    # Compute error
    error = np.sqrt((agent_x - estimated_pose["x"])**2 +
                     (agent_z - estimated_pose["z"])**2)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    
    ax.set_xlim((min_x_axis, max_x_axis))
    ax.set_ylim((min_z_axis, max_z_axis))
    ax.set_title(f'Step {step} | Localization Error: {error:.3f}m', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for particle weights
    cbar = plt.colorbar(scatter, ax=ax, label='Particle Weight')
    
    plt.tight_layout()
    # plt.savefig(f'localization_step_{step:03d}.png')
    # assert False
    # Convert plot to image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image

def ensure_output_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or f"{args.scene_name}"
    output_dir = args.output_root + '/' + run_name + '/' + args.trajectory_file.replace('.pkl', '')
    os.makedirs(output_dir, exist_ok=True)
    # output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_bounding_box(cfg, pcd):
    # if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
    #     try:
    #         return pcd.get_oriented_bounding_box(robust=True)
    #     except RuntimeError as e:
    #         print(f"Met {e}, use axis aligned bounding box instead")
    #         return pcd.get_axis_aligned_bounding_box()
    # else:
    return pcd.get_axis_aligned_bounding_box()

def stack_pcd_bboxes(pcd_bboxes_list: List):
    pcd_bboxes = []
    for bbox in pcd_bboxes_list:
        if isinstance(bbox, o3d.geometry.OrientedBoundingBox) or \
            isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
            bbox = np.asarray(bbox.get_box_points())
        else:
            raise ValueError("Input list must contain bounding boxes")
        bbox = torch.from_numpy(bbox).float()  # (8, 3)
        pcd_bboxes.append(bbox)
    stacked_pcd_bboxes = torch.stack(pcd_bboxes, dim=0) # (N, 8, 3)
    # print('stacked_pcd_bboxes', stacked_pcd_bboxes.shape)
    return stacked_pcd_bboxes

def get_visible_areas(
    obj_positions: List[np.ndarray], #Sequence[Dict[str, Any]],
    reachable_array: np.ndarray,
    max_distance: float,
) -> List[int]:
    """Get indices of objects visible from given position within FOV and distance.
    
    Args:
        position: 2D positions [x, z]
        reachable_positions: 2D positions [x, z]
        max_distance: maximum visible distance
    
    """
    visible_areas = []
    num_rays = 72
    default_obj_range = 1
    tolerance = 0.5
    max_distance = 4
    # obj_positions = list()
    visible_areas_sizes = list()
    visible_areas = list()

    for pos in obj_positions:
        if pos is None:
            print('pos is None')
            # visible_areas_sizes.append(0)
            # visible_areas.append(np.array([]))
            continue
        step = max(max_distance / 50.0, 0.1)
        angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False, dtype=np.float32)
        ray_dirs = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
        current = np.asarray(pos, dtype=np.float32)[[0, 2]]  # x, z
        visited_indices: set[int] = set()
        for direction in ray_dirs:
            traveled = 0.0
            while traveled <= max_distance:
                traveled += step
                ray_point = current + direction * traveled
            # Cast a coarse ray through the reachable grid to approximate line-of-sight visibility
                dists = np.linalg.norm(reachable_array - ray_point, axis=1)
                nearest_idx = int(np.argmin(dists))
                # print('traveled', traveled, 'dist', dists[nearest_idx])
                if dists[nearest_idx] <= tolerance:
                    if nearest_idx not in visited_indices:
                        visited_indices.add(nearest_idx)
                        # visible_total += 1
                elif traveled <= default_obj_range:
                    continue
                else:
                    break
        visible_areas_sizes.append(len(visited_indices))
        # visible_areas_indices.append(list(visited_on_ray))
        visible_areas.append(reachable_array[list(visited_indices)]
                             if len(visited_indices) > 0 else np.array([]))
    
    
    # print('visible_areas', visible_areas)
    # assert False
    return visible_areas, visible_areas_sizes

def minmax_norm(data: np.ndarray):
    min_val = np.min(data)
    max_val = np.max(data)
    norm_data = (data - min_val) / (max_val - min_val + 1e-6)
    return norm_data