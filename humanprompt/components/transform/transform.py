from typing import Dict, Union


class Transform:
    @staticmethod
    def transform(x: Union[str, Dict], y: Union[str, Dict] = None) -> str:
        if x and y:
            return f"Q: {x}\n" f"A: {y}"
        elif x:
            return f"Q: {x}"
        else:
            raise ValueError("x must be provided.")