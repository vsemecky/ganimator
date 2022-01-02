from . import functions
from .IDriver import IDriver
from .StyleGanTfDriver import StyleGanTfDriver
from .StyleGanPyDriver import StyleGanPyDriver
from .StyleGanPyExperimentalDriver import StyleGanPyExperimentalDriver
from .clips import *

import os
import sys

ganimator_dir = os.path.dirname(os.path.abspath(__file__))
print("ganimator_dir", ganimator_dir)
sys.path.append(ganimator_dir)
