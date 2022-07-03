
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import numpy as np
import cv2
import random
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode
from PIL import Image
import os
import random

def crop_object(image, box):
  """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
  """

  x_top_left = box[0]
  y_top_left = box[1]
  x_bottom_right = box[2]
  y_bottom_right = box[3]
  x_center = (x_top_left + x_bottom_right) / 2
  y_center = (y_top_left + y_bottom_right) / 2

  crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
  return crop_img

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.WEIGHTS = "C:/GitProject/CCP-Image-Analyser/model_final.pth"
cfg.DATASETS.TEST = ("deepfashion_val4", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

meta_data = MetadataCatalog.get("deepfashion_val").set(thing_classes=['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
            'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
            'long_sleeved_dress', 'vest_dress', 'sling_dress'])


path_dir = u'C:\\Users\\BrawnyClover\\Desktop\\2021CCP\\DeepFashion\\crawled\\casual'
result_dir = u'C:/Users/BrawnyClover/Desktop/2021CCP/DeepFashion/result'
files = os.listdir(path_dir)

for file in files:
    
    target_img = os.path.join(path_dir, file)
    print(target_img)
    img_array = np.fromfile(target_img, np.uint8)
    im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    outputs = predictor(im)
    # We can use `Visualizer` to draw the predictions on the image.
    print(outputs["instances"].pred_classes) 
    for metas in outputs["instances"].pred_classes:
        print(meta_data.thing_classes[metas])
    print(outputs["instances"].scores)

    v = Visualizer(im[:, :, ::-1], metadata=meta_data, scale=1.2)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    for idx in range(0, len(outputs["instances"].pred_classes)):
        
        box = outputs["instances"].pred_boxes[idx].tensor.numpy()[0]
        
        image = np.array(crop_object(Image.fromarray(im), box))
        file_name = os.path.splitext(file)[0]
        test_result = os.path.join(result_dir, file_name+'+'+meta_data.thing_classes[outputs["instances"].pred_classes[idx]]+'.jpg')
        cv2.imwrite(test_result, image)

