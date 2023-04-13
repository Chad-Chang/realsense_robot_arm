# --------------------------------- 공통 import --------------------------------- #

import time
import threading
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# --------------------------------- 재총 import --------------------------------- #

import pyrealsense2 as rs
import socket

# --------------------------------- 수환 import --------------------------------- #

import copy
from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
from data import cfg, set_cfg, set_dataset
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
from multiprocessing.pool import ThreadPool
from queue import Queue

# --------------------------------- 도혁 import --------------------------------- #
import serial
from collections import deque


# --------------------------------- 공통 초기 setting --------------------------------- #
flag = 0
start = 0

# --------------------------------- 수환 초기 setting --------------------------------- #

# np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.5f}".format(x)}) #numpy배열 소수점 출력 설정
# global img_numpy, update_check, inf_check, object_num, object_class, object_mask, class_num, class_set, mask_window_name
# update_check = False
# inf_check = False
# object_num = 0
# class_num = 0
# class_set = set()
# mask_window_name = set()

global img_numpy, update_check, inf_check, copy_strawberry, copy_leaf, main_number
update_check = False
inf_check = False
copy_strawberry = []
copy_leaf = []
main_number = 0

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

# --------------------------------- 재총 초기 setting --------------------------------- #

main_frame = 0
stop = False
point = (400, 300)
centerBx=200
centerBy=150
centerSx=200
centerSy=150
centerJx=200
centerJy=150
gooo = 0
plz_get = False
lengx=[0 for row in range(20)]
lengy=[0 for row in range(20)]
lengz=[0 for row in range(20)]
ori_y=[[0 for col in range(2)] for row in range(20)]
ori_z=[[0 for col in range(2)] for row in range(20)]

leng_total = 1
pointB = [centerBx, centerBy]
pointS = [centerSx, centerSy]
pointJ = [centerJx, centerJy]

distanceB = [0 for row in range(20)]
distanceS=[0 for row in range(20)]

send_berry = False

stem_x=[0 for row in range(20)]
stem_y=[0 for row in range(20)]
berry_x=[0 for row in range(20)]
berry_y=[0 for row in range(20)]
jullgi_x=[0 for row in range(20)]
jullgi_y=[0 for row in range(20)]

TCP_IP = '192.168.0.86'
TCP_PORT = 2002
BUFFER_SIZE = 512

mat = socket.socket(socket.AF_INET, socket.SOCK_STREAM)##
mat.connect((TCP_IP, TCP_PORT))##
stop_car = False

pipeline = rs.pipeline()
config = rs.config()

def nothing(x):
    pass

def show_distance(event, x, y, args, params):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold, distance
    global point
    point = (x, y)

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 10)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 6)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

# --------------------------------- 도혁 초기 setting --------------------------------- #
np.set_printoptions(threshold=np.inf, linewidth=np.inf) # 리스트를 제한없이 보기 위해 필요

# 로봇의 폭, 길이 / 라이다의 가로 세로 최대 영역 / 안전거리 설정


robot_width = 350
robot_length = 500
x_max = 1600
y_max = 500
safety_distance = 200

# 시리얼 통신 포트 설정
ser1 = serial.Serial(port='COM30', baudrate=128000)  # 허브 윗 칸, 전방 라이다
ser2 = serial.Serial(port='COM5', baudrate=128000)  # 허브 아랫 칸, 후방 라이다
ser3 = serial.Serial(port='COM6', baudrate=115200)  # ARDUINO

# 라이다 데이터 수신을 위한 정보 전송
ser1.isOpen()
ser2.isOpen()
values = bytearray([int('a5', 16), int('60', 16)])
ser1.write(values)
ser2.write(values)

# 제어를 위한 전역변수 초기화
check = 0
stop_val = 0
prevTime = 0  # Loop time 측정을 위함


# --------------------------------- 공통 function --------------------------------- #


# --------------------------------- 수환 function --------------------------------- #
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/yolact_resnet50_strawberry_31999_1600000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=10, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default='2', type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=5, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.55, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed) #


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    global img_numpy, update_check, inf_check, copy_strawberry, copy_leaf, main_number

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv.LINE_AA)

    # if num_dets_to_consider == 0:
    #     return img_numpy

    object_num = num_dets_to_consider                                                       # 검출된 객체의 수
    object_name = [cfg.dataset.class_names[classes[i]] for i in range(0, object_num)]       # 검출된 객체의 이름 = class 구분 X
    object_mask = masks.cpu().numpy().copy()                                                # 검출된 객체의 mask = class 구분 X
    strawberry = []                                                                         # 딸기만 구분해 저장할 변수
    leaf = []                                                                               # 잎만 구분해 저장할 변수
    strawberry_number = 0
    if args.display_text or args.display_bboxes:
        # 검출된 객체의 정보(이름, mask)가 각각의 list에 있기 때문에 동일 index끼리 묶고 list로 전환
        object_list = list(zip(classes, object_name, object_mask, boxes))
        # print(object_list)
        j = 0

        # 객체의 class별로 분류하는 코드
        # box 랑 text 및 score 출력
        for i in object_list:
            score = scores[j]
            if i[0] == 0: # strawberry
                strawberry_number += 1
                x1_strawberry, y1_strawberry, x2_strawberry, y2_strawberry = i[3]
                strawberry.append(i)

                # text_str = '%s: %.2f' % ('berry', score) if args.display_scores else object_name[j]
                text_str = '%.2f' % score
                font_face = cv.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1_strawberry, y1_strawberry - 3)
                text_color = [255, 255, 255]

                cv.rectangle(img_numpy, (x1_strawberry, y1_strawberry), (x2_strawberry, y2_strawberry), (0,0,255), thickness=2)
                cv.rectangle(img_numpy, (x1_strawberry, y1_strawberry), (x1_strawberry + text_w, y1_strawberry - text_h - 4), (0, 0, 255), -1)
                cv.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,cv.LINE_AA)

            if i[0] == 1: # leaf
                x1_leaf, y1_leaf, x2_leaf, y2_leaf = i[3]
                leaf.append(i)

                # text_str = '%s: %.2f' % (object_name[j], score) if args.display_scores else object_name[j]
                text_str = '%.2f' % score
                font_face = cv.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1_leaf, y1_leaf - 3)
                text_color = [255, 255, 255]

                cv.rectangle(img_numpy, (x1_leaf, y1_leaf), (x2_leaf, y2_leaf), (255, 0, 0), thickness=1)
                cv.rectangle(img_numpy, (x1_leaf, y1_leaf), (x1_leaf + text_w, y1_leaf - text_h - 4), (255, 0, 0), -1)
                cv.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv.LINE_AA)

            j += 1

    main_number = strawberry_number
    if update_check:
        copy_strawberry = strawberry.copy()  # 딸기의 정보만 있는 변수
        copy_leaf = leaf.copy()              # 잎의 정보만 있는 변수
        inf_check = True
        # update_check = False

    return img_numpy


def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()

    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]


def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)

    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                       'use_yolo_regressors', 'use_prediction_matching',
                       'train_masks']

        output = {
            'info': {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)


def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()


def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()


def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections: Detections = None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h * w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes, gt_boxes = split(gt_boxes)
                crowd_masks, gt_masks = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop,
                                                    score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h * w).cuda()
        boxes = boxes.cuda()

    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i, :], box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i, :, :], mask_scores[i])
            return

    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue

                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.
    Source: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


def evalimage(net: Yolact, path: str, save_path: str = None):
    global img_numpy
    frame = torch.from_numpy(cv.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv.imwrite(save_path, img_numpy)


def evalimages(net: Yolact, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'):
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def evalvideo(net: Yolact, path: str, out_path: str = None):
    global main_frame
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()

    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True

    if is_webcam:
        vid = cv.VideoCapture(int(path))
    else:
        vid = cv.VideoCapture(path)

    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps = round(vid.get(cv.CAP_PROP_FPS))
    frame_width = round(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv.CAP_PROP_FRAME_COUNT))

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    if target_fps ==0 :
        target_fps =1
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < args.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that
    def play_video():
        global main_frame, stop
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None:
                        main_frame = frame_buffer.get()
                        cv.imshow('main', main_frame)

                    else:
                        out.write(frame_buffer.get())
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                              % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                # This is split because you don't want savevideo to require cv display functionality (see #197)
                if out_path is None and cv.waitKey(1) == 27:
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001  # Let's just subtract a millisecond to be safe

                if out_path is None or args.emulate_playback:
                    # This gives more accurate timing than if sleeping the whole amount at once
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    # Let's not starve the main thread, now
                    time.sleep(0.001)
        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()

    extract_frame = lambda x, i: (
    x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None: print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None

            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args = [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)

                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in
                                          range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)

                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence) - 1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0

            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (
            fps, video_fps, frame_buffer.qsize())
            if not args.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')

    cleanup_and_exit()


def evaluate(net: Yolact, dataset, train_mode=False):
    global img_numpy
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            evalvideo(net, inp, out)
        else:
            evalvideo(net, args.video)
        return

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    if not args.display and not args.benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))

    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

                # Test flag, do not upvote
                if cfg.mask_proto_debug:
                    with open('scripts/info.txt', 'w') as f:
                        f.write(str(dataset.ids[image_idx]))
                    np.save('scripts/gt.npy', gt_masks)

                batch = Variable(img.unsqueeze(0))
                if args.cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                preds = net(batch)
            # Perform the meat of the operation here depending on our mode.
            if args.display:
                img_numpy = prep_display(preds, img, h, w)
            elif args.benchmark:
                prep_benchmark(preds, h, w)
            else:
                prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)

            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())

            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy)
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
            elif not args.no_bar:
                if it > 1:
                    fps = 1 / frame_times.get_avg()
                else:
                    fps = 0
                progress = (it + 1) / dataset_size * 100
                progress_bar.set_val(it + 1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), it + 1, dataset_size, progress, fps), end='')

        if not args.display and not args.benchmark:
            print()
            if args.output_coco_json:
                print('Dumping detections...')
                if args.output_web_json:
                    detections.dump_web()
                else:
                    detections.dump()
            else:
                if not train_mode:
                    print('Saving data...')
                    with open(args.ap_data_file, 'wb') as f:
                        pickle.dump(ap_data, f)

                return calc_map(ap_data)
        elif args.benchmark:
            print()
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000 * avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


def eval_thread():
    global img_numpy
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)


# --------------------------------- 도혁 function --------------------------------- #
def draw_line(gradient, intercept, n): # 직선 그리기
    x = []
    y = []
    for i in range(0, 2 * n):
        x.append(i - n)
        y.append((i - n) * gradient + intercept)

    return x, y


# Least Square Method 사용 : 기울기, Y절편 Return
def make_line(x, y, n):
    if n <= 1:
        gradient = 0
        intercept_y = 0

    else:
        sum_x = 0
        sum_y = 0
        sum_xy = 0
        sum_x_square = 0
        number = 0

        for angle in range(0, n):
            sum_x = sum_x + x[angle]
            sum_y = sum_y + y[angle]
            sum_xy = sum_xy + x[angle] * y[angle]
            sum_x_square = sum_x_square + x[angle] * x[angle]
            number = number + 1

        if (n * sum_x_square - (sum_x * sum_x)) != 0:
            gradient = ((number * sum_xy) - (sum_x * sum_y)) / (n * sum_x_square - (sum_x * sum_x))
            intercept_y = (sum_y / number) - (gradient * sum_x / number)
        else:
            gradient = 0
            intercept_y = 0

    return gradient, intercept_y


# BFS Algorithm 사용 : 군집을 대표 하는 점 Return, 이때 dis 는 군집 내 에서의 최대 거리
def make_a_point(x_data, y_data, size_x, size_y, dis):
    visited = [[0 for _ in range(size_y)] for _ in range(size_x)]
    dot_data = []
    count = []
    center_x = []
    center_y = []
    label = 1
    for x, y in zip(x_data, y_data):
        dot_data.append((x, y))
    # print(dot_data)
    for r, c in dot_data:
        if r >= size_x or c >= size_y:
            continue
        if visited[r][c] == 0:
            cnt = 1  # 카운트 초기화
            q = deque()
            q.append((r, c))
            visited[r][c] = label
            cen_x, cen_y = r, c  # 중심 초기값
            while q:
                x, y = q.popleft()
                for dx, dy in dot_data:
                    if (x == dx) and (y == dy):  # 자기 자신 패스
                        continue
                    if dx >= size_x or dy >= size_y:
                        continue
                    if (abs(x - dx) <= dis) and (abs(y - dy) <= dis) and (visited[dx][dy] == 0):
                        visited[dx][dy] = label
                        q.append((dx, dy))
                        cnt += 1
                        cen_x += dx
                        cen_y += dy
            count.append(cnt)

            if cnt >= 2:
                center_x.append(cen_x // cnt)  # 정수 값으로 반환
                center_y.append(cen_y // cnt)
                label += 1

    return center_x, center_y


# 전방/후방 라이다 선택
def choose_lidar(space, dist, max_x, max_y):
    global robot_width, robot_length

    x = [0 for _ in range(360)]
    y = [0 for _ in range(360)]
    x2 = [0 for _ in range(360)]
    y2 = [0 for _ in range(360)]
    x_dist = max_x - robot_length / 2
    y_dist = max_y - robot_width / 2
    if space == 'f':
        # 각도와 거리에 대한 값을 x,y 좌표로 변환
        for angle in range(0, 360):
            x[angle] = dist[angle] * math.cos(math.radians(angle))
            y[angle] = dist[angle] * math.sin(math.radians(angle)) * -1
            # 지정 거리 이내의 거리에 대해서 만 분석, 지정 거리 보다 먼 거리는 0으로 변경
            if (x[angle] <= 0 and y[angle] <= 0) or x[angle] > x_dist or x[angle] < -(robot_length + x_dist) or y[
                angle] > y_dist or y[angle] < -robot_width:
                x[angle] = 0
                y[angle] = 0
                x2[angle] = 0
                y2[angle] = 0
            else:
                x[angle] = int(x[angle] + robot_length / 2)
                y[angle] = int(y[angle] + robot_width / 2)
                x2[angle] = x[angle] + max_x
                y2[angle] = y[angle] + max_y

    else:
        for angle in range(0, 360):
            x[angle] = dist[angle] * math.cos(math.radians(angle)) * -1
            y[angle] = dist[angle] * math.sin(math.radians(angle))
            # 지정 거리 이내의 거리에 대해서 만 분석, 지정 거리 보다 먼 거리는 0으로 변경
            if (x[angle] >= 0-20 and y[angle] >= 0-20) or x[angle] > x_dist + robot_length or x[angle] < -x_dist or y[
                angle] > robot_width or y[angle] < -y_dist:
                x[angle] = 0
                y[angle] = 0
                x2[angle] = 0
                y2[angle] = 0
            else:
                x[angle] = int(x[angle] - robot_length / 2)
                y[angle] = int(y[angle] - robot_width / 2)
                x2[angle] = x[angle] + max_x
                y2[angle] = y[angle] + max_y

    new_x, new_y = make_a_point(x2, y2, max_x * 2, max_y * 2, 25)

    return x, y, new_x, new_y


def plot_lidar(space, dist, max_x, max_y):
    global robot_width, robot_length

    x_draw = []
    y_draw = []

    x, y, new_x, new_y = choose_lidar(space, dist, max_x, max_y)

    for k in range(len(new_x)):
        new_x[k] = new_x[k] - max_x
        new_y[k] = new_y[k] - max_y
        if (new_x[k] != 0) and (new_y[k] != 0) and (new_x[k] != -max_x):
            x_draw.append(new_x[k])
            y_draw.append(new_y[k])
        else:
            continue
    gradient, intercept_y = make_line(x_draw, y_draw, len(x_draw))

    return x, y, new_x, new_y, gradient, intercept_y


def send2ard(ard, data):
    data = data.encode('utf-8')
    ard.write(data)


def point_plot(a, b, c, d, e, f, g, h):
    plt.fill_between([robot_length / 2, -robot_length / 2, -robot_length / 2, robot_length / 2],
                     [robot_width / 2, robot_width / 2, -robot_width / 2, -robot_width / 2], alpha=0.5)
    plt.scatter(a, b, c='r', s=8)
    plt.scatter(c, d, c='b', s=8)
    plt.scatter(e, f, c='r', s=8)
    plt.scatter(g, h, c='b', s=8)


# 다음 블록 찾는 함수
def find_next(x_2, y_2):
    cnt_1 = 0
    cnt_2 = 0

    for i in range(len(x_2)):
        if (x_2[i] < 500) and (x_2[i] > 400) and (y_2[i] < 0):
            cnt_1 += 1
        if (x_2[i] > -500) and (x_2[i] < -300) and (y_2[i] < 0):
            cnt_2 += 1

    # print('cnt1 : ', cnt_1)
    # print('cnt2 : ', cnt_2)
    if (cnt_1 >= 1) and (cnt_2 >= 1):
        return 1  # 다음 블록 찾음
    else:
        return 0  # 아직 더 찾아야 함


# 일정 거리를 기준으로 정지하는 함수
def safety_stop(x, y):
    for i in range(len(x)):
        if (x[i] > 0) and (x[i] < safety_distance + robot_length/2) and (y[i] < robot_width/2) and (y[i] > -robot_width/2):
            return 1
        else:
            return 0


def rotate_stop(x, y):
    rot_num = 0
    for i in range(len(x)):
        if (x[i] > 0) and (x[i] < 900 + robot_length / 2) and (y[i] < robot_width/2) and (y[i] > -robot_width/2):
            rot_num += 1
        else:
            pass
    print('rot_num : ', rot_num)
    if rot_num > 20:
        return 1
    else:
        return 0


def canny(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (0, 0), 1)
    cny = cv.Canny(blur, 50, 150)
    return cny


def _checksum(data):
    try:
        ocs = _HexArrToDec((data[6], data[7]))
        lsn = data[1]
        cs = 0x55AA ^ _HexArrToDec((data[0], data[1])) ^ _HexArrToDec((data[2], data[3])) ^ _HexArrToDec(
            (data[4], data[5]))
        for i in range(0, 2 * lsn, 2):
            cs = cs ^ _HexArrToDec((data[8 + i], data[8 + i + 1]))

        if cs == ocs:
            return True
        else:
            return False
    except Exception as e:
        return False


def _HexArrToDec(data):
    littleEndianVal = 0
    for i in range(0, len(data)):
        littleEndianVal = littleEndianVal + (data[i] * (256 ** i))
    return littleEndianVal


def _AngleCorr(dist):
    if dist == 0:
        return 0
    else:
        return math.atan(21.8 * ((155.3 - dist) / (155.3 * dist))) * (180 / math.pi)


def _Calculate(d):
    ddict = []
    lsn = d[1]
    Angle_fsa = ((_HexArrToDec((d[2], d[3])) >> 1) / 64.0)
    Angle_lsa = ((_HexArrToDec((d[4], d[5])) >> 1) / 64.0)
    if Angle_fsa < Angle_lsa:
        Angle_diff = Angle_lsa - Angle_fsa
    else:
        Angle_diff = 360 + Angle_lsa - Angle_fsa
    for i in range(0, 2 * lsn, 2):
        dist_i = _HexArrToDec((d[8 + i], d[8 + i + 1])) / 4
        Angle_i_tmp = ((Angle_diff / float(lsn)) * (i / 2)) + Angle_fsa
        if Angle_i_tmp > 360:
            Angle_i = Angle_i_tmp - 360
        elif Angle_i_tmp < 0:
            Angle_i = Angle_i_tmp + 360
        else:
            Angle_i = Angle_i_tmp

        Angle_i = Angle_i + _AngleCorr(dist_i)
        ddict.append((dist_i, Angle_i))
    return ddict


def _Mean(data):
    length_of_data_without_zero = sum([i != 0 for i in data])
    if len(data) > 0 and length_of_data_without_zero != 0:
        #        return int(sum(data)/len(data)) # original By ydlidar
        return float(sum(data) / length_of_data_without_zero)  # modified for remove zero value
    return 0


def code(ser):
    data1 = ser.read(6000)
    data2 = data1.split(b"\xaa\x55")[1:-1]
    distdict = {}
    for i in range(0, 360):
        distdict.update({i: []})
    for i, e in enumerate(data2):
        try:
            if e[0] == 0:
                if _checksum(e):
                    d = _Calculate(e)
                    for ele in d:
                        angle = math.floor(ele[1])
                        if (angle >= 0) and (angle < 360):
                            distdict[angle].append(ele[0])
        except Exception as e:
            pass
    for i in distdict.keys():
        distdict[i] = _Mean(distdict[i])
    yield distdict


# --------------------------------- 공통 Thread --------------------------------- #


# --------------------------------- 수환&재총 Thread --------------------------------- #
class CoordinateThread(threading.Thread):

    def run(self) -> None:

        global plz_get, update_check, flag, gooo, send_berry, stop, start,stem_x,stem_y,berry_x,berry_y,jullgi_y,jullgi_x

        cv.namedWindow('key', cv.WINDOW_NORMAL)  # 키보드 입력을 위한 window창

        while True:
            if cv.waitKey(1) & 0xFF == ord('s'):
                update_check = True

            if cv.waitKey(1) & 0xFF == ord('z'):
                flag = 1

            if cv.waitKey(1) & 0xFF == ord('x'):
                flag = 2

            if cv.waitKey(1) & 0xFF == 27:
                cv.destroyWindow('key')



            frames_s = pipeline.wait_for_frames()
            color_frame = frames_s.get_color_frame()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames_s)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image

            # Validate that both frames are valid

            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            grey_color = 153
            depth_image_3d = np.dstack(
                (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels

            # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, get_next_frame)
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
            # images = np.hstack((bg_removed, depth_colormap))

            if inf_check:
                if start == 1 and stop == False:
                    send_berry = True
                else :
                    send_berry = False
                # update_check = False
                if main_number != 0:
                    # 마스크 확인 하기 위한 부분 : 모든 마스크를 윈도우창에 띄움
                    # for i in range(len(copy_strawberry)):
                    #     cv.imshow(copy_strawberry[i][1] + str(i), copy_strawberry[i][2])
                    # for i in range(len(copy_leaf)):
                    #     cv.imshow(copy_leaf[i][1] + str(i), copy_leaf[i][2])

                    # copy_strawberry 와 copy_leaf = class번호, class이름, mask, box좌표 순으로 저장된 변수
                    # --------------------마스크 중점 계산을 위해 먼저 640, 480 크기짜리 텐서 만드는 부분-----------------------#
                    row_vector = []  # X좌표 계산 위한 배열
                    column_vector = []  # Y좌표 계산 위한 배열
                    for i in range(640):
                        row_vector.append([i])
                        if i <= 479:
                            column_vector.append([i])
                    row_vector = np.array(row_vector)
                    column_vector = np.array(column_vector)

                    strawberry_mid_list = []
                    leaf_mid_list = []

                    # 딸기 중점 계산
                    for i in copy_strawberry:
                        strawberry_area = i[2].sum()
                        strawberry_x = i[2].sum(axis=0)
                        strawberry_y = i[2].sum(axis=1)
                        total_strawberry_x = np.multiply(strawberry_x, row_vector).sum()
                        total_strawberry_y = np.multiply(strawberry_y, column_vector).sum()
                        strawberry_mid_x = int(total_strawberry_x / strawberry_area)
                        strawberry_mid_y = int(total_strawberry_y / strawberry_area)
                        strawberry_mid_dot = (strawberry_mid_x, strawberry_mid_y)
                        strawberry_mid_list.append(strawberry_mid_dot)
                        cv.circle(img_numpy, strawberry_mid_dot, 5, (0, 0, 255), -1)

                    # 잎 중점 계산
                    for i in copy_leaf:
                        leaf_area = i[2].sum()
                        leaf_x = i[2].sum(axis=0)
                        leaf_y = i[2].sum(axis=1)
                        total_leaf_x = np.multiply(leaf_x, row_vector).sum()
                        total_leaf_y = np.multiply(leaf_y, column_vector).sum()

                        if leaf_area != 0:
                            leaf_mid_x = int(total_leaf_x / leaf_area)
                            leaf_mid_y = int(total_leaf_y / leaf_area)
                        else:
                            leaf_mid_x = 0
                            leaf_mid_y = 0

                        leaf_mid_dot = (leaf_mid_x, leaf_mid_y)
                        leaf_mid_list.append(leaf_mid_dot)
                        cv.circle(img_numpy, leaf_mid_dot, 5, (0, 255, 0), -1)

                    # 2차 발표
                    # 중점 인식
                    # berry_mask = cv.cvtColor(copy_strawberry[0][2], cv.COLOR_GRAY2RGB)
                    # cv.circle(berry_mask, strawberry_mid_list[0], 7, (0, 0, 255), -1)
                    # leaf_mask = cv.cvtColor(copy_leaf[0][2], cv.COLOR_GRAY2RGB)
                    # cv.circle(leaf_mask, leaf_mid_list[0], 7, (0, 255, 0), -1)
                    # cv.imshow('mask_berry', berry_mask)
                    # cv.imshow('mask_leaf', leaf_mask)

                    # 즁점 긋기
                    # for i in range(len(copy_strawberry)):
                    #     cv.imshow(copy_strawberry[i][1] + str(i), copy_strawberry[i][2])
                    # for i in range(len(copy_leaf)):
                    #     cv.imshow(copy_leaf[i][1] + str(i), copy_leaf[i][2])
                    # test_img0 = img_numpy.copy()
                    # test_img1 = img_numpy.copy()
                    # test_img2 = img_numpy.copy()
                    # test_img3 = img_numpy.copy()
                    # cv.line(test_img0, strawberry_mid_list[0], leaf_mid_list[0], (0, 0, 0), thickness=3)
                    # cv.line(test_img1, strawberry_mid_list[0], leaf_mid_list[1], (0, 0, 0), thickness=3)
                    # cv.line(test_img2, strawberry_mid_list[1], leaf_mid_list[0], (0, 0, 0), thickness=3)
                    # cv.line(test_img3, strawberry_mid_list[1], leaf_mid_list[1], (0, 0, 0), thickness=3)
                    # cv.imshow('test0', test_img0)
                    # cv.imshow('test1', test_img1)
                    # cv.imshow('test2', test_img2)
                    # cv.imshow('test3', test_img3)

                    # 딸기와 잎 매칭
                    match_strawberry = []
                    match_leaf = []
                    for i in strawberry_mid_list:
                        standard_distance = 1000
                        standard_angle = 25
                        j_prev = (0, 0)
                        for j in leaf_mid_list:
                            distance_now = math.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2)
                            angle_now = math.atan2(i[1] - j[1], j[0] - i[0]) * 180 / math.pi
                            angle_minus = abs(90 - angle_now)
                            if 65 <= angle_now <= 115:
                                if angle_minus <= standard_angle:
                                    standard_angle = angle_minus
                                    if distance_now <= standard_distance:
                                        standard_distance = distance_now
                                        if i not in match_strawberry:
                                            match_strawberry.append(i)
                                        if j_prev in match_leaf:
                                            match_leaf.remove(j_prev)
                                            leaf_mid_list.append(j_prev)
                                        if j not in match_leaf:
                                            j_prev = j
                                            match_leaf.append(j)
                                            leaf_mid_list.remove(j)
                    for i in range(len(match_strawberry)):
                        cv.line(img_numpy, match_strawberry[i], match_leaf[i], (0, 0, 0), thickness=3)
                        distanceB[i] = depth_image[match_strawberry[i][1], match_strawberry[i][0]]
                        distanceS[i] = depth_image[match_leaf[i][1], match_leaf[i][0]]

                        if match_leaf[i][0] - match_strawberry[i][0] == 0:
                            ori_z[i] = 1
                        else:
                            ori_z[i] = math.atan2((match_strawberry[i][1] - match_leaf[i][1]),
                                                  (match_strawberry[i][0] - match_leaf[i][0]))
                        depth_stem = depth_image[match_leaf[i][1], match_leaf[i][0]]
                        depth_stem_next = depth_image[match_leaf[i][1], match_leaf[i][0]]

                        stem_x[i] = int(match_leaf[i][0])  # 822
                        stem_y[i] = int(match_leaf[i][1])
                        berry_x[i] = int(match_strawberry[i][0])
                        berry_y[i] = int(match_strawberry[i][1])
                        jullgi_x[i] = int(match_leaf[i][0] + (match_leaf[i][0] - match_strawberry[i][0]))
                        jullgi_y[i] = int(match_leaf[i][1] + (match_leaf[i][1] - match_strawberry[i][1]))
                        #print(i,berry_x[i])
                        if berry_x[i] > 400:
                            flag =1
                            send2ard(ser3, 'a')
                            time.sleep(3)
                            start = 1
                        # if berry_x




                        take_low = 0

                        # print(j, stem_x, stem_y, depth_stem_next)
                        cv.circle(main_frame, (berry_x[i], berry_y[i]), 2, (0, 255, 0), 2)  # 822
                        cv.circle(main_frame, (jullgi_x[i], jullgi_y[i]), 2, (0, 255, 0), 2)
                        cv.circle(main_frame, (stem_x[i], stem_y[i]), 2, (0, 0, 255), 2)
                        #print(distanceB[i])
                        grey_color = 153
                        depth_image_3d = np.dstack(
                            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
                        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color,
                                              main_frame)

                        if plz_get == True and gooo == 0 and send_berry == True and distanceB[i] != 0 :  # 822
                            stop == True
                            lengx[i] = (distanceB[i]) * math.tan(math.pi * 22.68 / 180) * (
                                    240 - stem_y[i]) / 240  # +30*math.sin(ori_z)
                            lengy[i] = (distanceB[i]) * math.tan(math.pi * 24.61 / 180) * (
                                    stem_x[i] - 320) / 320  # +30*math.cos(ori_z)
                            # lengx2 = (distanceS) * math.tan(math.pi * 22.68 / 180) * (240 - centerSy) / 240# + 60
                            # lengy2 = (distanceS) * math.tan(math.pi * 25.11 / 180) * (centerSx - 320) / 320
                            lengz[i] = distanceB[i] - 80
                            data = str(lengx[i]) + '/' + str(lengy[i]) + '/' + str(lengz[i]) + '/' + str(
                                ori_z[i]) + '/' + str(ori_y[i]) + '/'
                            data = data.encode()
                            mat.send(data)
                            gooo = 1
                            plz_get = False
                        if plz_get == True and gooo == 1 and send_berry == True:
                            stop = True
                            lengx[i] = (100) * math.tan(math.pi * 22.68 / 180) * (
                                    240 - centerJy) / 240 + 75  # +30*math.sin(ori_z)
                            lengy[i] = (100) * math.tan(math.pi * 24.61 / 180) * (
                                    centerJx - 320) / 320 +25 # -10 +30*math.cos(ori_z)# lengx2 = (distanceS) * math.tan(math.pi * 22.68 / 180) * (240 - centerSy) / 240# + 60
                            # lengy2 = (distanceS) * math.tan(math.pi * 25.11 / 180) * (centerSx - 320) / 320
                            lengz[i] = 50
                            data = str(lengx[i]) + '/' + str(lengy[i]) + '/' + str(lengz[i]) + '/' + str(
                                ori_z[i]) + '/' + str(ori_y[i]) + '/'
                            data = data.encode()
                            mat.send(data)
                            gooo = 0
                            plz_get = False


                        # lengx2 = (distanceS) * math.tan(math.pi * 22.68 / 180) * (240 - centerSy) / 240# + 60
                        # lengy2 = (distanceS) * math.tan(math.pi * 25.11 / 180) * (centerSx - 320) / 320

                        # if stop_car == True :
                        #     time.sleep(1)
                        #     send_berry = True


                    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
                    #                images = np.hstack((bg_removed, depth_colormap))
                    distance = depth_image[point[1], point[0]]
                    cv.putText(main_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv.FONT_HERSHEY_PLAIN,
                                2,
                                (0, 0, 0), 2)
                    depth_stem = depth_image[centerSy, centerSx]
                    depth_stem_next = depth_image[centerSy, centerSx]

                    # cv2.circle(main_frame, (centerBx, centerBy), 2, (0, 0, 255), 10)
                    # cv2.circle(main_frame, (centerSx, centerSy), 2, (0, 0, 255), 10)


                    #print(match_strawberry, match_leaf)
                    # not_match_index = []
                    # for i in range(len(strawberry_mid_list)):
                    #     if strawberry_mid_list[i] not in match_strawberry:
                    #         not_match_index.append(i)
                    # #print(not_match_index)
                    #
                    # for i in not_match_index:
                    #     img_ellipse = copy_strawberry[i][2].astype('uint8')
                    #     contours, _ = cv.findContours(img_ellipse, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                    #     _, triangle = cv.minEnclosingTriangle(contours[0])
                    #     a1 = tuple(triangle[0][0])
                    #     a2 = tuple(triangle[1][0])
                    #     a3 = tuple(triangle[2][0])
                    #     cv.line(img_1, (int(a1[0]), int(a1[1])), (int(a2[0]), int(a2[1])), (0, 0, 0), thickness=5)
                    #     cv.line(img_1, (int(a1[0]), int(a1[1])), (int(a3[0]), int(a3[1])), (0, 0, 0), thickness=5)
                    #     cv.line(img_1, (int(a2[0]), int(a2[1])), (int(a3[0]), int(a3[1])), (0, 0, 0), thickness=5)
                    #     if a1[1] <= a2[1] <= a3[1]:
                    #         dot = (int(a3[0]), int(a3[1]))
                    #     elif a1[1] <= a3[1] <= a2[1]:
                    #         dot = (int(a2[0]), int(a2[1]))
                    #     else:
                    #         dot = (int(a1[0]), int(a1[1]))
                    #     new_dot = (strawberry_mid_list[i][0] + abs(strawberry_mid_list[i][0] - dot[0]), strawberry_mid_list[i][1] - abs(strawberry_mid_list[i][1] - dot[1]))
                    #     cv.line(img, strawberry_mid_list[i], new_dot, (0, 255, 0), thickness=5)
                    #     ellipse = cv.fitEllipse(contours[0])
                    #     ellipse_degree = ellipse[2]
                    #     mid_x = int(ellipse[0][0])
                    #     mid_y = int(ellipse[0][1])
                    #     long_axis = ellipse[1][0]
                    #     line_x = mid_x + int(long_axis / 2 * math.cos(math.radians(ellipse_degree - math.pi)))
                    #     line_y = mid_y - int(long_axis / 2 * math.sin(math.radians(ellipse_degree - math.pi)))
                    #     cv.line(img, (mid_x, mid_y), (line_x, line_y), (0, 0, 0), thickness=5)
                    #     cv.ellipse(copy_strawberry[i][2], ellipse, (0, 255, 0), 2)
                    #     cv.putText(img, str(ellipse_degree), (mid_x, mid_y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                elif main_number == 0:
                    print('\r Strawberry is nothing', end='')
                    # update_check = False
                    # inf_check = False
                    start = 0
                    flag = 0
                    send2ard(ser3, 'b')

                    stem_x = [0 for row in range(20)]
                    stem_y = [0 for row in range(20)]
                    berry_x = [0 for row in range(20)]
                    berry_y = [0 for row in range(20)]
                    jullgi_x = [0 for row in range(20)]
                    jullgi_y = [0 for row in range(20)]
# --------------------------------- 매트랩 통신 Thread --------------------------------- #
class commu(threading.Thread):
    global gooo, plz_get

    while True :
        if plz_get == False :
            data = mat.recv(BUFFER_SIZE)
            msg = data.decode()
            if msg == 'he' :
                msg = 0
                plz_get = True

# --------------------------------- 도혁 Thread --------------------------------- #

# 라이다 쓰레드
class LidarThread(threading.Thread):

    def run(self) -> None:
        global flag, check, stop_val, start

        while True:

            angle_data1 = code(ser1)
            angle_data2 = code(ser2)
            plt.figure(1, figsize=(12, 4))
            plt.cla()
            x1, y1, new_x1, new_y1, gradient1, intercept1 = plot_lidar('f', next(angle_data1), x_max, y_max)
            x2, y2, new_x2, new_y2, gradient2, intercept2 = plot_lidar('b', next(angle_data2), x_max, y_max)
            plt.xlim(-x_max, x_max)
            plt.ylim(-y_max, y_max)
            point_plot(x1, y1, new_x1, new_y1, x2, y2, new_x2, new_y2)
            draw_x1, draw_y1 = draw_line(gradient1, intercept1, 800)
            draw_x2, draw_y2 = draw_line(gradient2, intercept2, 800)
            plt.plot(draw_x1, draw_y1, 'g', draw_x2, draw_y2, 'g')
            plt.pause(0.001)

            #if flag == 1:
             #   send2ard(ser3, 'a')

            #else:
             #   send2ard(ser3, 'b')

            # send2ard(ser3, data)
            # time.sleep(0.1)


# 카메라 쓰레드 (다음 블록을 찾을 때에만 사용)
class CameraThread(threading.Thread):

    def run(self) -> None:
        global flag, stop_val
        height, width = 540, 960
        x_cen, y_cen, x_left, gap = 480, 230, 415, 130
        # f1 = m*(x-x_cen)-y+y_cen
        # f2 = -m*(x-x_cen)-y+y_cen
        m = int(y_cen/(x_cen-x_left))
        y1 = -m*x_cen + y_cen
        y2 = m*x_cen + y_cen

        cv.namedWindow('Camera')
        cap = cv.VideoCapture(0)
        print('Camera ready')


        while True:
            ret, frame = cap.read()
            image = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
            canny_image = canny(image)
            hough_image = cv.cvtColor(canny_image, cv.COLOR_GRAY2BGR)  # canny image 에 빨, 파, 초 색깔을 표현하기 위함

            # threshold(만나는 점의 기준) = 50, minLineLength(선의 최소 길이) = 20, maxLineGap(선과 선 사이의 최대 허용간격) = 30
            lines = cv.HoughLinesP(canny_image, 1, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=30)

            check_num = 0
            if lines is not None:
                for i in range(0, len(lines)):
                    li = lines[i][0]

                    if abs(li[1]-li[3]) < 10:  # 수평에 가까운 직선만 검출
                        li[1] = height - li[1]
                        li[3] = height - li[3]  # 좌표계 변환

                        if (li[0] < x_cen) and (li[2] < x_cen):  # 좌측 점 2개
                            if ((m * li[0] - y1) > li[1]) or ((m * li[2] - y1) > li[3]):
                                check_num += 1
                                cv.line(hough_image, (li[0], 540 - li[1]), (li[2], 540 - li[3]), (0, 0, 255), 2,
                                        cv.LINE_AA)
                        elif (li[0] < x_cen) and (li[2] > x_cen):  # 좌측 점 1개, 우측 점 1개
                            if ((m * li[0] - y1) > li[1]) or ((-m * li[2] + y2) > li[3]):
                                check_num += 1
                                cv.line(hough_image, (li[0], 540 - li[1]), (li[2], 540 - li[3]), (255, 0, 0), 2,
                                        cv.LINE_AA)
                        elif (li[0] > x_cen) and (li[2] > x_cen):  # 우측 점 2개
                            if ((-m * li[0] + y2) > li[1]) or ((-m * li[2] + y2) > li[3]):
                                check_num += 1
                                cv.line(hough_image, (li[0], 540 - li[1]), (li[2], 540 - li[3]), (0, 255, 0), 2,
                                        cv.LINE_AA)
                        else:
                            pass

                    else:
                        pass

            if check_num == 0:  # 삼각형 내부를 통과하는 직선이 1개도 검출되지 않은 경우
                cv.putText(image, "STOP", (420, 400), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), thickness=3)
                stop_val = 1

            else:
                stop_val = 0

            cv.imshow("Camera", hough_image)
            # cv.imshow("hough", hough_image)

            if cv.waitKey(1) & 0xff == 27:
                break

        cap.release()
        cv.destroyAllWindows()


class FlagThread(threading.Thread):

    def run(self) -> None:

        global flag

        while True:
            print('| flag :', flag, ' | start :', start, ' | update check :', update_check, ' | inf_chack :', inf_check, ' | main_number', main_number,)
            print(' | berry : ',berry_x[0], 'depth : ', distanceB[0])
            time.sleep(3)

# --------------------------------- main Thread --------------------------------- #


thread_eval = threading.Thread(target=eval_thread, args=()) # Learning Thread
thread_eval.start()
t1 = LidarThread()
# t2 = CameraThread()
t3 = CoordinateThread()
t4 = FlagThread()
t5 = commu()
t1.start()
# t2.start()
t3.start()
t4.start()
t5.start()

