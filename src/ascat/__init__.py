import pkg_resources
import sys
sys.path.insert(1,'../src')

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'

from src.ascat.read_native.cdr import *
from src.ascat.h_saf import *
