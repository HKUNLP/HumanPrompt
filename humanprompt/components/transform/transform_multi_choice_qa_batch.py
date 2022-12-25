from typing import Any, Dict, Union, List

from .transform_base import Transform


class MultiChoiceBatchTransform(Transform):
    @staticmethod
    def transform(
            x: List[Dict], y: List[Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform x and y into a multi-choice format.

        Args:
            x: a dict with keys "question" and "choices", where "choices" is a list of labels and text.
            y: a dict with keys "answer" and maybe others.
            **kwargs: other arguments

        Returns: a string of question, choices, and answer

        """
        if not isinstance(x[0], Dict) \
                or (y and not isinstance(y[0], Dict)):
            raise TypeError("x and y should be dict in multi-choice task.")

        transformed = ""
        for idx, x_ in enumerate(x, 1):
            transformed += f"Q[{idx}]: {x_['question']}\n"
            transformed += "Answer choices[{}]: {}\n".format(
                idx,
                " ".join(
                    [
                        "({}) {}".format(label.lower(), text.lower())
                        for label, text in zip(
                        x_["choices"]["label"], x_["choices"]["text"]
                    )
                    ]
                )
            )
        transformed += "A[1]: "
        # for idx, x_ in enumerate(x, 1):
        #     transformed += f"Q: {x_['question']}\n"
        #     transformed += "Answer choices: {}\n".format(
        #         " ".join(
        #             [
        #                 "({}) {}".format(label.lower(), text.lower())
        #                 for label, text in zip(
        #                 x_["choices"]["label"], x_["choices"]["text"]
        #             )
        #             ]
        #         )
        #     )
        # transformed += "A: "

        if y:
            # TODO
            pass

        return transformed
