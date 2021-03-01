import pkg_resources
import sys
sys.path.insert(1,'../src')

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'

from ascat.read_native.cdr import *
from ascat.h_saf import *
