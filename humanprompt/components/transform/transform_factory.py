from typing import Type

from .transform_base import Transform
from .binder import BinderTransform
from .cot import CoTTransform
from .db_text2sql import DBText2SQLTransform
from .qa import QATransform
from .zero_shot_cot import ZeroShotCoTTransform

from hub.cot.commonsense_qa.transform_cot_commonsense_qa import CoTCommonsenseQATransform


class TransformFactory(object):
    """
    Provide built factory class for creating transforms.
    """

    current_transforms = {
        "default": Transform,
        "qa": QATransform,
        "cot": CoTTransform,
        "zero_shot_cot": ZeroShotCoTTransform,
        "binder": BinderTransform,
        "db_text2sql": DBText2SQLTransform,
        # Method&Dataset-specific
        "cot-commonsense_qa": CoTCommonsenseQATransform
        # Add more transforms here
    }

    @staticmethod
    def get_transform(transform: str) -> Type[Transform]:
        # If the transform is in current_transforms, return the identity class
        if transform in TransformFactory.current_transforms:
            return TransformFactory.current_transforms[transform]
        else:
            try:
                # If the transform is not in current_transforms, try to import it from the transform path
                from pydoc import locate

                transform_class = locate(transform)
                return transform_class  # type: ignore
            except Exception:
                # If the transform is not in, treat it as a class name and return the class
                return globals()[transform]
