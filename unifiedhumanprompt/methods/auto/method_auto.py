from collections import OrderedDict

from ..cot.method import CoTMethod
from ..standard.method import Method
from .auto_factory import BaseAutoMethod

METHOD_MAPPING_NAMES = OrderedDict(
    [
        # TODO: Method class definition should have different names for different methods
        ("standard", Method),
        ("cot", CoTMethod),
    ]
)


class AutoMethod(BaseAutoMethod):
    _method_mapping = METHOD_MAPPING_NAMES
