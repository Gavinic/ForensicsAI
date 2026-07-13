import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

from .biomConv_detect import BiomConvDetector
from .biomVit_detect import BiomVitDetector
from .clip_detector import CLIPDetector
from .idbmVit_detect import IdbmVitDetector

# from .altfreezing_detector import AltFreezingDetector
from .orth_detector import OrthDetector
from .replk_detector import ReplkDetector
