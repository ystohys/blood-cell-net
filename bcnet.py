from torch import nn
import torchvision.models.detection._utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2 ,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights
)

def get_bcnet(model_name, num_classes, num_trainable_layers=1):
    if model_name == 'frcnn':
        # function on next line returns a pre-trained faster RCNN model
        bcnet = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            trainable_backbone_layers=num_trainable_layers
            )
        # Replacing "head" network with a new, untrained one
        # Done by looking at the source code for the original fasterRCNN class
        in_features = bcnet.roi_heads.box_predictor.cls_score.in_features
        bcnet.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
            )
    
    elif model_name == 'retina':
        # function on next line returns a pre-trained RetinaNet model
        bcnet = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=num_trainable_layers
        )
        # Replacing "head" network with a new, untrained one
        # Done by looking at source code for original RetinaNet class
        in_features = bcnet.backbone.out_channels
        bcnet.head = RetinaNetHead(
            in_features, 
            bcnet.anchor_generator.num_anchors_per_location()[0],
            num_classes
            )
    elif model_name == 'ssd':
        # function on next line returns a pre-trained SSD model
        bcnet = ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT,
            trainable_backbone_layers=num_trainable_layers
        )
        # Replacing "head" network with a new, untrained one
        # Done by looking at source code for original SSD class
        if hasattr(bcnet.backbone, "out_channels"):
            in_features = bcnet.backbone.out_channels
        else:
            in_features = det_utils.retrieve_out_channels(bcnet.backbone, (300,300))
        bcnet.head = SSDHead(
            in_features,
            bcnet.anchor_generator.num_anchors_per_location(),
            num_classes
        )
    return bcnet