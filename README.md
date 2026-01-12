# Open-vacabulary Scene Graph Construction and Monte-Carlo Localization 
Developed by Chang Chen@hku, 2026.1.12

## 0. Content
This repo was folked from [concept-graph](scene_graphs/3dvproject/conceptgraph/localization). I made three main differences: 
 - [conceptgraph/localization](./conceptgraph/localization/): contains all the necessary codes for running a simple monte-carlo localization when kidnappig a robot into the scene. This code requires building scene graphs and doing relocalization on an offline collected RGBD datatset, e.g., Replica dataset, and AI2THOR (I mainly used), or a rosbag in a real-world scene. Several improvement can be made on it, e.g., modifying it to advanced localization algorithms, or adding an active localization strategy. This code uses Grounded-SAM + CLIP for semantic understanding using the NVIDIA RTX 4090 GPU. More details of the original concept-graph can be found in [REAMDE_conceptgraph.md](./README_conceptgraph.md).
 - [scene_understand](./scene_understand/): contains an open-vocabulary object detector using Detic + CLIP, designing for real-world deployment with its better inference speed than Grounded-SAM + CLIP. I have custimized the Detic for better integration with other modules. In experiments, it concumed 1.3s per frame on NVIDIA Jetson Orin NX. 
 - [dynamic_scene_graph](./dynamic_scene_graph/): contains a code draft for establishing a dynamic topological graph with the networkx library. I haven't complete it, and it would be useful to build on it.

Install repo with submodules:
```bash
git clone git@github.com:chch9907/scene-graph-localization.git --recurse-submodules
```

## 1. Setup for conceptgraph-localization

The env variables needed can be found in `env_vars.bash.template`. When following the setup guide below, you can duplicate that files and change the variables accordingly for easy setup. 

### Install the required libraries

We recommend setting up a virtual environment using virtualenv or conda. Our code has been tested with Python 3.10.12. It may also work with other later versions. We also provide the `environment.yml` file for Conda users. In generaly, directly installing conda env using `.yml` file may cause some unexpected issues, so we recommand setting up the environment by the following instructions and only using the `.yml` file as a reference. 

Sample instructions for `conda` users. 

```bash
conda create -n conceptgraph anaconda python=3.10
conda activate conceptgraph

# Install the required libraries
pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy

# for yolo
pip install ultralytics

# Install the Faiss library (CPU version should be fine)
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

##### Install Pytorch according to your own setup #####
# For example, if you have a GPU with CUDA 11.8 (We tested it Pytorch 2.0.1)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
# conda install pytorch3d -c pytorch3d # This detects a conflict. You can use the command below, maybe with a different version
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2

# Install the gradslam package and its dependencies
# Please clone and install them in separate folders, not within the concept-graphs folder. 
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
pip install .
cd ..
git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout conceptfusion
pip install .
```

### Install [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) package

Follow the instructions on the original [repo](https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker). ConceptGraphs has been tested with the codebase at this [commit](https://github.com/IDEA-Research/Grounded-Segment-Anything/commit/a4d76a2b55e348943cba4cd57d7553c354296223). Grounded-SAM codebase at later commits may require some adaptations. (**If facing ModuleNotFoundError: No module named 'torch' when compiling grounding-dino, try running python3 setup.py install**)

First checkout the package by 

```bash
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
```

Then, install the package Following the commands listed in the original GitHub repo. You can skip the `Install osx` step and the "optional dependencies". 

During this process, you will need to set the `CUDA_HOME` to be where the CUDA toolkit is installed. 
The CUDA tookit can be set up system-wide or within a conda environment. We tested it within a conda environment, i.e. installing [cudatoolkit-dev](https://anaconda.org/conda-forge/cudatoolkit-dev) using conda. 

```bash
# i.e. You can install cuda toolkit using conda
conda install -c conda-forge cudatoolkit-dev

# and you need to replace `export CUDA_HOME=/path/to/cuda-11.3/` by 
export CUDA_HOME=/path/to/anaconda3/envs/conceptgraph/
```

You also need to download `ram_swin_large_14m.pth`, `groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth` (and optionally `tag2text_swin_14m.pth` if you want to try Tag2Text) following the instruction [here](https://github.com/IDEA-Research/Grounded-Segment-Anything#label-grounded-sam-with-ram-or-tag2text-for-automatic-labeling). 

After installation, set the path to Grounded-SAM as an environment variable

```bash
export GSA_PATH=/path/to/Grounded-Segment-Anything
```

### Install this repo

```bash
git clone git@github.com:concept-graphs/concept-graphs.git
cd concept-graphs
pip install -e .
```

## 2. Prepare dataset

### Replica dataset (optional)

ConceptGraphs takes posed RGB-D images as input. Here we show how to prepare the dataset using [Replica](https://github.com/facebookresearch/Replica-Dataset) as an example. Instead of the original Replica dataset, download the scanned RGB-D trajectories of the Replica dataset provided by [Nice-SLAM](https://github.com/cvg/nice-slam). It contains rendered trajectories using the mesh models provided by the original Replica datasets. 

Download the Replica RGB-D scan dataset using the downloading [script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) in [Nice-SLAM](https://github.com/cvg/nice-slam#replica-1) and set `$REPLICA_ROOT` to its saved path.

```bash
export REPLICA_ROOT=/path/to/Replica

export CG_FOLDER=/path/to/concept-graphs/
export REPLICA_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/dataconfigs/replica/replica.yaml
```

### AI2Thor-related experiments (what I used)

Use our own [fork](https://github.com/georgegu1997/ai2thor), where some changes were made to record the interaction trajectories. 

```bash
cd .. # go back to the root folder CFSLAM
git clone git@github.com:georgegu1997/ai2thor.git
cd ai2thor
git checkout main5.0.0
pip install -e .

# This is for the ProcThor dataset.
pip install ai2thor-colab prior --upgrade
```


## 3. Generating AI2Thor datasets

1. Use `$AI2THOR_DATASET_ROOT` as the directory ai2thor dataset and save it to a variable. Also set the scene used from AI2Thor. 

```bash
cd ./conceptgraph
AI2THOR_DATASET_ROOT=/home/user/ldata/ai2thor  # your dataset directory
SCENE_NAME=train_3  # any available scene 
```

1. Generate a densely captured grid map for the selected scene. 
```bash
# Uniform sample camera locations (XY + Yaw) 
python scripts/generate_ai2thor_dataset.py --dataset_root $AI2THOR_DATASET_ROOT --scene_name $SCENE_NAME --sample_method uniform --n_sample -1 --grid_size 0.5
```


## 4. Extract 2D (Detection) Segmentation and per-resgion features

First, (Detection) Segmentation results and per-region CLIP features are extracted. In the following, we provide two options. 
* The first one (ConceptGraphs) uses SAM in the "segment all" mode and extract class-agnostic masks. 
* The second one (ConceptGraphs-Detect) uses a tagging model and a detection model to extract class-aware bounding boxes first, and then use them as prompts for SAM to segment each object. 

```bash
cd ./conceptgraph
SCENE_NAME=train_3  # any available scene 
AI2THOR_DATASET_ROOT=/home/user/ldata/ai2thor  # your dataset directory
AI2THOR_CONFIG_PATH=dataset/dataconfigs/ai2thor/ai2thor.yaml

# The CoceptGraphs (without open-vocab detector, I used this cmd) 
python scripts/generate_gsa_results.py \
    --dataset_root $AI2THOR_DATASET_ROOT \
    --dataset_config $AI2THOR_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set none \
    --stride 5

# The ConceptGraphs-Detect 
CLASS_SET=ram
python scripts/generate_gsa_results.py \
    --dataset_root $AI2THOR_DATASET_ROOT \
    --dataset_config $AI2THOR_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride 5 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses
```

The above commands will save the detection and segmentation results in `$REPLICA_ROOT/$SCENE_NAME/`. 
The visualization of the detection and segmentation can be viewed in `$REPLICA_ROOT/$SCENE_NAME/gsa_vis_none` and `$REPLICA_ROOT/$SCENE_NAME/gsa_vis_ram_withbg_allclasses` respectively. 


## 5. Run the 3D object mapping system
The following command builds an object-based 3D map of the scene, using the image segmentation results from above.  

* Use `save_objects_all_frames=True` to save the mapping results at every frame, which can be used for animated visualization by `scripts/animate_mapping_interactive.py` and `scripts/animate_mapping_save.py`. 
* Use `merge_interval=20  merge_visual_sim_thresh=0.8  merge_text_sim_thresh=0.8` to also perform overlap-based merging during the mapping process. 

```bash
cd ./conceptgraph
SCENE_NAME=train_3  # any available scene of AI2THOR
AI2THOR_DATASET_ROOT=/home/user/ldata/ai2thor  # your dataset directory
AI2THOR_CONFIG_PATH=dataset/dataconfigs/ai2thor/ai2thor.yaml


# Using the CoceptGraphs (without open-vocab detector, class-agnostic)
THRESHOLD=1.2
python slam/cfslam_pipeline_batch.py \
    dataset_root=$AI2THOR_DATASET_ROOT \
    dataset_config=$AI2THOR_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.95 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=none \
    class_agnostic=True \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8

# or the ConceptGraphs-Detect (class-aware)
THRESHOLD=1.2
python slam/cfslam_pipeline_batch.py \
    dataset_root=$AI2THOR_DATASET_ROOT \
    dataset_config=$AI2THOR_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=ram_withbg_allclasses \
    skip_bg=False \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1
```

The above commands will save the mapping results in `$REPLICA_ROOT/$SCENE_NAME/pcd_saves`. It will create two `pkl.gz` files, where the one with `_post` suffix indicates results after some post processing, which we recommend using.


## 6. Visualize the object-based mapping results

```bash
python scripts/visualize_cfslam_results.py --result_path /path/to/output.pkl.gz
```

Then in the open3d visualizer window, you can use the following key callbacks to change the visualization. 
* Press `b` to toggle the background point clouds (wall, floor, ceiling, etc.). Only works on the ConceptGraphs-Detect.
* Press `c` to color the point clouds by the object class from the tagging model. Only works on the ConceptGraphs-Detect.
* Press `r` to color the point clouds by RGB. 
* Press `f` and type text in the terminal, and the point cloud will be colored by the CLIP similarity with the input text. 
* Press `i` to color the point clouds by object instance ID. 

## 7. Topological scene graph construction (TODO)
Concetp-graph used [build_scenegraph_cfslam](./conceptgraph/scenegraph/build_scenegraph_cfslam.py) to construct the scene graph, but required gpt4 and llava to predict the object spatial relationship. We only need a simple and useful scene graph builder. I put a code draft in the [dynamic_scene_graph](./dynamic_scene_graph/dynamic_scene_graph.py), but I haven't complete it. It would be helpful for development based on this code draft or the code in conceptgraph. For localization, I simply cluster the objects and judge the connectivity based on the Euclidean distance.


## 8. Monte-Carlo localization with prebuilt scene graph
My developed codes inlcude: monte_carlo_localization.py, semantic_detection.py, semantic_particle_filter.py,localization_utils.py, vis_results.py.

The MCL runs by first randomly sampling a 40-step trajectory within the reachable positions (read from reachable.json) to control the robot. Each pose of the trajectory in the AI2THOR is defined as (x, y, z, yaw), where XYZ follows the left-hand rule, where Z is the viewing direction, X is right, and Y is up. Euler angles are in degrees. At each step, the MCL estimates the robot pose viaa the standard prediction-update-resample pipeline, while instead matching the CLIP-based visual embeddings with those obseved at each particle pose to update the likelihood.


The following command runs different versions of the MCL. The --trajectory_file specifies the trajectory loading for controlling the robot and conducting the experiment. If its file does not exist, the code will first randomly sample a trajectory. The --exp_name specifies different modes of the information weights multiplying on each matched similarity. All the related codes are in the [semantic_particle_filter.py](./conceptgraph/localization/semantic_particle_filter.py#L391) (update function). In practice, I found my designed information weights cannot produce stable improvement compared to the baseline in different trials. So it is recommended to use baseline mode. 


```bash
# (default) At each update step, when successfully matching the detected objects and the objects on the scene graph, assign each matched object with a uniform weight. It can achieve stable convergence in different trails, though the accuracy may not be so good.
python3 ./localization/monte_carlo_localization.py --trajectory_file trajectory1.pkl # --exp_name baseline

# Semantic uniqueness: prioritize the less common objects with a recipocal of the TF-IDF score.
python3 ./localization/monte_carlo_localization.py --trajectory_file trajectory1.pkl --exp_name semantic --alpha 1  

# Cluster centraility: prioritize the objects that are the centroid of an object cluster, measured by the degrees.
python3 ./localization/monte_carlo_localization.py --trajectory_file trajectory1.pkl --exp_name central --alpha 0  

# Combination with a balancing alpha
python3 ./localization/monte_carlo_localization.py --trajectory_file trajectory1.pkl --exp_name combine0.5 --alpha 0.5

# visualize the results, specifying the labels to compare different methods
python3 ./localization/vis_results.py --log-dir loc_outputs/train_3/trajectory1 --labels baseline semantic central
```


## 9. Setup for Detic
Follow the installation instruction of [Detic](./scene_understand/Detic/README.md#L27).


## 10. Others
```bash
# install torch, torchvision with cu118
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# use huggingface mirror
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com
```