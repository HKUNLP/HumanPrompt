from typing import Type

from .base import Transform
from .binder import BinderTransform
from .cot import CoTTransform
from .qa import QATransform
from .zero_shot_cot import ZeroShotCoTTransform


class TransformFactory(object):
    """
    Provide built factory class for creating transforms.
    """

    current_transforms = {
        "default": Transform,
        "qa": QATransform,
        "cot": CoTTransform,
        "zero_shot_cot": ZeroShotCoTTransform,
        "binder": BinderTransform
        # Add more transforms here
    }

    @staticmethod
    def get_transform(transform: str) -> Type[Transform]:
        # If the transform is in current_transforms, return the identity class
        if transform in TransformFactory.current_transforms:
            return TransformFactory.current_transforms[transform]
        else:
            # If the transform is not in, treat it as a class name and return the class
            return globals()[transform]
