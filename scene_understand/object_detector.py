# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
from PIL import Image
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Detic'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Detic/third_party/CenterNet2/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# print(sys.path)
from centernet.config import add_centernet_config  # type: ignore
from detic.config import add_detic_config  # type: ignore
from detic.predictor import VisualizationDemo  # type: ignore

#! modify line 23 in src/scene_understand/Detic/detic/config.py:
#! _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
#!        'Detic/datasets/metadata/lvis_v1_train_cat_info.json'


# custom_vocabulary = ['Access_Panel', 'Accessible_Toilet', 'Auditorium_Feature', 'Auditorium_Seating', 'Basic_Wall', 'Basin_Array', 'Basketball_Court', 'Basketball_Hoop', 'Beam', 'Ceiling', 'Ceiling_Reflector', 'Chair_Arm', 'Channel', 'Concrete_Filling', 'Connector', 'Curtain_Pelmet', 'Curtain_Wall_Door_Double_Storefront', 'Curtain_Wall_Door_Single_Glass', 'Domestic_Toilet', 'Door', 'Double_Door', 'Electric_Door', 'Elevator', 'Escalator', 'Fire_Shutter', 'Floor', 'Floor_Slab', 'Folding_Partition_Wall', 'Glass_Double_Door', 'Hose_Reel_Door', 'Ladder', 'Locker', 'Louver', 'Metal_Door', 'Mirror', 'Box', 'Opening', 'Panel', 'Parking_Space', 'Perforated_Panel_Brace', 'Piano', 'Ping_Pong_Table', 'Plinth', 'Railing', 'Railing_Trellis', 'Ramp', 'Rectangular_Column', 'Rectangular_Steel_Column', 'Round_Column', 'Security_Shutter', 'Shower_Curtain', 'Single_Basin', 'Sink', 'Sink_Vanity_Square', 'Sliding_Door_Single', 'Smoke_Vent', 'Stage_Wall', 'Stair', 'Telescopic_Seating', 'Toilet_Cubicle', 'Trolley_Track', 'Vanity_Unit_Basin', 'Wall_Hung_Urinal', 'Water_Meter_Cabinet', 'Water_Tank_Cover', 'Window']

custom_vocabulary = ['Basketball_Court', 'Basketball_Hoop', 'Chair_Arm', 'Channel', 'Connector', 'Toilet', 'Door','Elevator', 'Escalator', 'Fire_Shutter', 'Ladder', 'Locker', 'Louver', 'Mirror', 'Box', 'Panel', 'Parking_Space', 'Piano', 'Ping_Pong_Table', 'Railing', 'Rectangular_Column', 'Round_Column', 'Security_Shutter', 'Shower_Curtain', 'Basin', 'Sink', 'Smoke_Vent', 'Stair', 'Telescopic_Seating', 'Vanity', 'Wall_Hung_Urinal', 'Water_Meter_Cabinet', 'Water_Tank_Cover', 'Window', 'safety barrier', 'trolley track']
custom_vocabulary = [name.replace('_', ' ') for name in sorted(custom_vocabulary)]
custom_vocabulary_str = ','.join(custom_vocabulary)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg



class VLM:
    def __init__(self, model_type, device):
        if model_type == 'clip':
            import clip
            model, preprocess = clip.load("ViT-B/32", device=device)
            self.image_encoder = model.encode_image
            self.text_encoder = model.encode_text
            self.preprocess = preprocess
            self.tokenize = clip.tokenize
        elif model_type == 'vild':
            raise NotImplementedError()
        else:
            raise ValueError()



class MyDetic(object):
    def __init__(self, args, vlm_type='clip'):
        global custom_vocabulary, custom_vocabulary_str
        if args.vocabulary == 'custom' and not len(args.custom_vocabulary):
            args.custom_vocabulary = custom_vocabulary_str
            print('custom_vocabulary len:', len(custom_vocabulary))
        cfg = setup_cfg(args)
        self.args = args
        self.demo = VisualizationDemo(cfg, args)
        
        self.metadata = MetadataCatalog.get(str(time.time()))
        self.metadata.thing_classes = custom_vocabulary
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vlm = VLM(vlm_type, self.device)  # use clip to extract visual features from the boxes detected by Detic
        self.prompt = 'a '
        self.texts = self.vlm.tokenize([self.prompt + x for x in custom_vocabulary]).to(self.device)
        self.text_features = self.vlm.text_encoder(self.texts)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        print('text_features:', self.text_features.shape)
        
    
    def __call__(self, img, img_path, use_clip=False, save_output=True, verbose=False):
        start_time = time.time()
        raw_predictions = self.demo.simple_predict(img)
        if verbose:
            print(
                "{} in {:.2f}s".format(
                    "detected {} instances".format(len(raw_predictions["instances"]))
                    if "instances" in raw_predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
        pred_features = raw_predictions['instances'].pred_box_features.to(self.device)  
        pred_features = F.normalize(pred_features, dim=1, p=2)

        predictions_dict = {
            'bboxes': to_numpy(raw_predictions["instances"].pred_boxes.tensor).astype(np.int64),
            'scores': raw_predictions["instances"].scores.detach(),
            'labels': raw_predictions["instances"].pred_classes,
            'masks': raw_predictions["instances"].pred_masks,
            'features': pred_features
        }
        predictions_dict['regions'] = [img[yl:yr, xl:xr] 
                                  for (xl, yl, xr, yr) in predictions_dict['bboxes']
                                  ]
        
        if use_clip:
            # print('use clip for retrieval.')
            clip_labels = self.retrieval(predictions_dict, img)
            if len(clip_labels):
                print('labels|clip:', to_numpy(clip_labels), 'detic:', to_numpy(raw_predictions['instances'].pred_classes),
                    'compare:', to_numpy(clip_labels == raw_predictions['instances'].pred_classes))
            raw_predictions['instances'].pred_classes = clip_labels

        if save_output:
            v = Visualizer(img[:, :, ::-1], self.metadata)
            out = v.draw_instance_predictions(raw_predictions['instances'].to("cpu"))
            # out_path = Path(tempfile.mkdtemp()) / "out.png"
            out_filename = os.path.join(self.args.output, img_path.split('/')[-1])
            cv2.imwrite(out_filename, out.get_image()[:, :, ::-1])
            
        return predictions_dict
    
    
    def retrieval(self, predictions, img):
        regions = predictions['regions']
        labels = predictions['labels']
        regions_proc = []
        
        # regions_proc = [self.vlm.preprocess(Image.fromarray(reg, 'RGB')) for reg in regions]
        # img_proc = self.vlm.preprocess(Image.fromarray(img, 'RGB')).unsqueeze(0).to(self.device)
        # if not len(regions_proc):
        #     return torch.tensor([])
        if not len(regions):
            return predictions['labels']
        with torch.no_grad():
            ## use clip features
            # regions_proc = torch.stack(regions_proc, dim=0).to(self.device)
            # image_features = self.vlm.image_encoder(regions_proc)
            # image_features /= image_features.norm(dim=-1, keepdim=True)

            # use Detic features
            image_features = predictions['features']
            similarity = (100.0 * image_features @ self.text_features.type(torch.float32).T).softmax(dim=-1)
            argmax_sims = torch.argmax(similarity, dim=1)
            print(argmax_sims)
        return argmax_sims

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--use_clip", action='store_true')
    parser.add_argument("--save_output", action='store_true')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    mydetector = MyDetic(args)
    # vlm = VLM(model_type='clip')
    os.makedirs(args.output, exist_ok=True)
    print(vars(args))
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        # if os.path.isdir(args.input):
        #     args.input = [os.path.join(args.input, name) for name in sorted(os.listdir(args.input))]
        
        
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions = mydetector(img, path, args.use_clip, args.save_output, verbose=True)
            
            # if args.output:
            #     if os.path.isdir(args.output):
            #         assert os.path.isdir(args.output), args.output
            #         out_filename = os.path.join(args.output, os.path.basename(path))
            #     else:
            #         assert len(args.input) == 1, "Please specify a directory with args.output"
            #         out_filename = args.output
            #     visualized_output.save(out_filename)

