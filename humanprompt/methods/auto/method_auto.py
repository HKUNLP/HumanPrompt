from collections import OrderedDict

from ..cot.method import CoTMethod
from ..react.method import ReActMethod
from ..standard.method import StandardMethod
from ..zero_shot_cot.method import ZeroShotCoTMethod
from .auto_factory import BaseAutoMethod

METHOD_MAPPING_NAMES = OrderedDict(
    [
        # TODO: Method class definition should have different names for different methods
        ("standard", StandardMethod),
        ("cot", CoTMethod),
        ("zero_shot_cot", ZeroShotCoTMethod),
        ("react", ReActMethod),
    ]
)


class AutoMethod(BaseAutoMethod):
    _method_mapping = METHOD_MAPPING_NAMES
