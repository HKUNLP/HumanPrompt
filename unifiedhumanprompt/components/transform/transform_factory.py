from typing import Callable

from unifiedhumanprompt.components.transform.transform import Transform


class TransformFactory:

    transform_funcs = {
        'default': Transform.transform
    }

    @staticmethod
    def get_transform(template: str = 'default') -> Callable:
        # TODO: Add more transform templates, use a map to store them
        return TransformFactory.transform_funcs[template]
