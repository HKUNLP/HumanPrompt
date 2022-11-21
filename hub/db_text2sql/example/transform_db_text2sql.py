from typing import Any, Dict, Union

from humanprompt.components.transform.transform_base import Transform
from humanprompt.components.transform.transform_db import DBTransform


class DBText2SQLTransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict[str, Union[str, Dict]]],
        y: Union[str, Dict] = None,
        **kwargs: Any,
    ) -> str:
        """
        Transform for text2sql on db prompting,
        which is firstly introduced in Evaluating the "Text-to-SQL Capabilities of Large Language Models"
        (https://arxiv.org/abs/2204.00498).

        Args:
            x: The input to the model (a natural language query).
            y: The output of the model (a SQL query).
            **kwargs: Additional arguments to the model.

        Returns: The transformed input for text2sql on db prompting.

        """

        assert isinstance(x, Dict)
        assert isinstance(x["db"], str)

        # DB prompt: database information
        db_prompt = DBTransform.transform(x, **kwargs)

        # Question prompt: question information
        question_prompt = "-- Using valid SQLite, answer the following questions for the tables provided above.\n-- {}".format(
            x["question"]
        )

        # Answer prompt: answer information
        answer_prompt = ""
        if y is not None:
            assert isinstance(y, Dict)
            answer_prompt = y["query"] + "\n"

        return f"{db_prompt}\n{question_prompt}\n{answer_prompt}"
