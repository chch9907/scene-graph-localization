import os
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import open_clip
from localization_utils import compute_clip_features

try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e


GSA_PATH = '/home/user/HKUra/workspace/scene_graphs/Grounded-Segment-Anything'
    

# Disable torch gradient computation
torch.set_grad_enabled(False)

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    else:
        raise ValueError(variant)
    

class SemanticDetectors:
    def __init__(self, args, classes):
        print('Initialize Grounded-SAM model...')
        ### Initialize the Grounding DINO model ###
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
            device=args.device
        )
        self.sam_predictor = get_sam_predictor(args.sam_variant, args.device)

        # Initialize the CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        self.clip_model = self.clip_model.to(args.device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        self.classes = classes
        self.args = args
        print('Finished initializing Grounded-SAM model.')
    
    # Prompting SAM with detected boxes
    @staticmethod
    def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, 
                                       image: np.ndarray, 
                                       xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    def detect(self, image):
        # image = cv2.imread(color_path) # This will in BGR color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
        image_pil = Image.fromarray(image_rgb)
        
        # Using GroundingDINO to detect and SAM to segment
        detections = self.grounding_dino_model.predict_with_classes(
            image=image, # This function expects a BGR image...
            classes=self.classes,
            box_threshold=self.args.box_threshold,
            text_threshold=self.args.text_threshold,
        )
        image_crops, image_feats, text_feats, confidence = \
            np.array([]), np.array([]), np.array([]), np.array([])
        seg_masks = []
        if len(detections.class_id) > 0:
            ### Non-maximum suppression ###
            # print(f"Before NMS: {len(detections.xyxy)} boxes")
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                self.args.nms_threshold
            ).numpy().tolist()
            # print(f"After NMS: {len(detections.xyxy)} boxes")

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            
            # Somehow some detections will have class_id=-1, remove them
            valid_idx = detections.class_id != -1
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]

            # # Somehow some detections will have class_id=-None, remove them
            # valid_idx = [i for i, val in enumerate(detections.class_id) if val is not None]
            # detections.xyxy = detections.xyxy[valid_idx]
            # detections.confidence = detections.confidence[valid_idx]
            # detections.class_id = [detections.class_id[i] for i in valid_idx]


            ### Segment Anything ###
            if len(detections.class_id) > 0:
                detections.mask = self.get_sam_segmentation_from_xyxy(
                    sam_predictor=self.sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )
                seg_masks = np.array(detections.mask)
                confidence = np.array(detections.confidence)
                # Compute and save the clip features of detections  
                image_crops, image_feats, text_feats = compute_clip_features(
                    image_pil, detections, 
                    self.clip_model, 
                    self.clip_preprocess, 
                    self.clip_tokenizer, 
                    self.classes, 
                    self.args.device)

            
            ### Visualize results ###
            # annotated_image, labels = vis_result_fast(image, detections, classes)
            
            # save the annotated grounded-sam image
            # cv2.imwrite(vis_save_path, annotated_image)
        
        return image_crops, image_feats, text_feats, seg_masks, confidence