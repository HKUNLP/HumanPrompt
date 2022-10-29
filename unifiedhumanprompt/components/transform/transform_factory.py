from base import Transform
from qa import QATransform
from table_qa import TableQATransform
from cot import CoTTransform


class TransformFactory(object):
    """
    Provide built factory class for creating transforms.
    """
    current_transforms = {
        'base': Transform,
        'qa': QATransform,
        'cot': CoTTransform,
        'table_qa': TableQATransform,
        # Add more transforms here
    }

    @staticmethod
    def get_transform(transform: str) -> Transform:
        # If the transform is in current_transforms, return the identity class
        if transform in TransformFactory.current_transforms:
            return TransformFactory.current_transforms[transform]
        else:
            # If the transform is not in, treat it as a class name and return the class
            return globals()[transform]
