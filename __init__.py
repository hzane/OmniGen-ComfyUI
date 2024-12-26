import sys
import folder_paths
import os.path as osp
now_dir = osp.dirname(__file__)
sys.path.append(now_dir)

from .nodes import OmniGenNode
from .loader import OmniGenLoader

NODE_CLASS_MAPPINGS = {
    "OmniGenNode": OmniGenNode,
    "OmniGenLoader": OmniGenLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniGenNode": "OmniGen",
    "OmniGenLoader": "Load OmniGen Model"
}