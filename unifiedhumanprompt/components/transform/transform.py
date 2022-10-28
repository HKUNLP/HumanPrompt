class Transform():
    @staticmethod
    def transform(x, y=None):
        if y is None:
            return x
        else:
            return f"{x}\n{y}"
