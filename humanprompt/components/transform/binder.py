from typing import Any, Dict, Union

from binder.nsql.database import NeuralDB

from ..utils.db_utils import build_db_prompt
from .base import Transform


class BinderTransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict[str, Union[str, Dict]]],
        y: Union[str, Dict] = None,
        **kwargs: Any,
    ) -> str:
        """
        Transform for Binder (https://arxiv.org/abs/2210.02875),
        a method for generating "Binder"(a mixture of natural language and symbolic language based on SQL/Python etc.)
        queries from natural language.
        Args:
            x: The input to the model (a natural language query).
            y: The output of the model (a Binder query).
            **kwargs: Additional arguments to the model.

        Returns: The transformed input for binder prompting.
        """

        assert isinstance(x, Dict)
        assert isinstance(x["table"], Dict)

        db = NeuralDB(tables=[{"title": x["table"]["page_title"], "table": x["table"]}])

        table = db.get_table_df()
        title = db.get_table_title()

        # Head prompt: task instruction
        generate_type = kwargs["generate_type"]
        header_prompt = {
            "answer": """\n-- Answer the question based on the given table below.\n\n""",
            "nsql": """\n-- Parse the question into NeuralSQL based on the given table below.\n\n""",
            "sql": """\n-- Parse the question into SQL based on the given table below.\n\n""",
            "npython": """\n-- Parse the question into NeuralPython based on the given table below.\n\n""",
            "python": """\n-- Parse the question into Python based on the given table below.\n\n""",
        }.get(
            generate_type,
            """\n-- Generate NeuralSQL and question pairs based on the given table below.\n\n""",
        )

        # DB prompt: database information
        prompt_style = kwargs["prompt_style"] if "prompt_style" in kwargs else None
        db_prompt = (
            build_db_prompt(table, title, prompt_style)
            if prompt_style
            else build_db_prompt(table, title)
        )

        # QA prompt: question and answer
        question = x["question"]
        qa_prompt = {
            "answer": "Q: {}\nA: ".format(question),
            "nsql": "Q: {}\nNeuralSQL: ".format(question),
            "sql": "Q: {}\nSQL: ".format(question),
            "npython": "Q: {}\nNeuralPython: ".format(question),
            "python": "Q: {}\nPython: ".format(question),
        }.get(generate_type, "Q: {}\nA: ".format(question))

        if y is None:
            return f"{header_prompt}{db_prompt}{qa_prompt}"
        else:
            return (
                f"{db_prompt}{qa_prompt}{y['nsql']}"
                if isinstance(y, Dict)
                else f"{db_prompt}{qa_prompt}{y}"
            )
