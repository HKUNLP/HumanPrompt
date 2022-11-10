from collections import OrderedDict

from .auto_factory import BaseAutoMethod
from ..standard.method import StandardMethod
from ..cot.method import CoTMethod
from ..zero_shot_cot.method import ZeroShotCoTMethod
from ..react.method import ReActMethod
from ..binder.method import BinderMethod


METHOD_MAPPING_NAMES = OrderedDict(
    [
        # TODO: Method class definition should have different names for different methods
        ("standard", StandardMethod),
        ("cot", CoTMethod),
        ("zero_shot_cot", ZeroShotCoTMethod),
        ("react", ReActMethod),
        ("binder", BinderMethod),
    ]
)


class AutoMethod(BaseAutoMethod):
    _method_mapping = METHOD_MAPPING_NAMES
