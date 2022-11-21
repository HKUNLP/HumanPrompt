from typing import Type

from humanprompt.artifacts import HUB_SOURCE

from .transform_base import Transform
from .transform_db import DBTransform
from .transform_multi_choice_qa import MultiChoiceQATransform
from .transform_simple_qa import QATransform
from .transform_table import TableTransform


class TransformFactory(object):
    """
    Provide built factory class for creating transforms.
    """

    current_transforms = {
        # Only the fundamental transforms
        "default": Transform,
        "simple_qa": QATransform,
        "multi_choice_qa": MultiChoiceQATransform,
        "db": DBTransform,
        "table": TableTransform,
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

                transform_class = locate("{}.{}".format(HUB_SOURCE, transform))
                return transform_class  # type: ignore
            except Exception:
                # If the transform is not in, treat it as a class name and return the class
                return globals()[transform]
