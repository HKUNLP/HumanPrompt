class Transform:
    @staticmethod
    def transform(x, y=None):
        if x and y:
            return f"Q: {x}\n" f"A: {y}"
        elif x:
            return f"Q: {x}"
        else:
            raise ValueError("x must be provided.")
