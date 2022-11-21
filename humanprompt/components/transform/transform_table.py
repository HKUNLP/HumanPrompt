from typing import Any, Dict, Union

from .transform_base import Transform
from .transform_db import build_table_prompt


class TableTransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform the table into a prompt.

        Args:
            x: input, contains the key of 'table' and 'table_name'
            y: not required
            **kwargs:

        Returns: the prompt of database.

        """
        assert isinstance(x, Dict)

        db_prompt = build_table_prompt(
            x["table"],
            x["table_name"] if "table_name" in x else "table",
            kwargs["prompt_style"]
            if "prompt_style" in kwargs
            else "create_table_select_3",
        )
        return db_prompt
