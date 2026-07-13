import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import LOSSFUNC

from .am_softmax import AMSoftmax_OHEM, AMSoftmaxLoss
from .bce_loss import BCELoss
from .capsule_loss import CapsuleLoss
from .consistency_loss import ConsistencyCos
from .contrastive_regularization import ContrastiveLoss
from .cross_entropy_loss import CrossEntropyLoss
from .id_loss import IDLoss
from .js_loss import JS_Loss
from .l1_loss import L1Loss
from .patch_consistency_loss import PatchConsistencyLoss
from .supercontrast_loss import SupConLoss
from .vgg_loss import VGGLoss
