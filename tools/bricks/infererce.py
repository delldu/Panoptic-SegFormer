# from mmcv.runner import checkpoint
from mmdet.apis.inference import init_detector,LoadImage, inference_detector
import easymd
import pdb

# config = 'configs/panformer/panformer_r50_24e_coco_panoptic.py'
# checkpoints = "checkpoints/panoptic_segformer_r50_2x.pth"

config = "configs/panformer/panformer_pvtb5_24e_coco_panoptic.py"
checkpoints = "checkpoints/panoptic_segformer_pvtv2b5_2x.pth"

img = 'images/001.png'
model = init_detector(config, checkpoint=checkpoints)
results = inference_detector(model, img)


pdb.set_trace()

