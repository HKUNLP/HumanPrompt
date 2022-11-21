from typing import Any, Dict, Union


class Transform(object):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform x and y(may not) into a prompt.
        Args:
            x: input, could be a str or a dict, when it is a str, it is the input itself.
            y: output, could be a str or a dict, when it is a str, it is the output itself.
            **kwargs: other arguments

        Returns: a string of prompt
        """

        if not isinstance(x, str) and isinstance(y, str):
            raise UserWarning("x and y should be str when transform is not specified")

        if y is None:
            return f"{x}"
        else:
            return f"{x}\n{y}"
