class Transform(object):

    @staticmethod
    def transform(x, y=None, **kargs):
        """
        Transform x and y(may not) into a prompt.
        Args:
            x: input, could be a str or a dict, when it is a str, it is the input itself.
            y: output, could be a str or a dict, when it is a str, it is the output itself.

        Returns: a string of prompt

        """

        if not isinstance(x, str) and isinstance(y, str):
            raise UserWarning("x and y should be str when transform is not specified")

        if y is None:
            return f"{x}"
        else:
            return f"{x}\n{y}"
