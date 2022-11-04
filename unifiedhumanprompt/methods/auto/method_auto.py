from collections import OrderedDict

from .auto_factory import BaseAutoMethod
from ..standard.method import Method
from ..cot.method import CoTMethod


METHOD_MAPPING_NAMES = OrderedDict(
    [
        # TODO: Method class definition should have different names for different methods
        ('standard', Method),
        ('cot', CoTMethod)
    ]
)


class AutoMethod(BaseAutoMethod):
    _method_mapping = METHOD_MAPPING_NAMES
