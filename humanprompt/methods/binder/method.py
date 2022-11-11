import importlib.util
import os
from typing import Any, Dict, List, Optional, Union

from ...components.post_hoc import HocPoster
from ...components.prompt import PromptBuilder
from ...methods.base_method.method import PromptMethod


def is_binder_available() -> bool:
    return importlib.util.find_spec("binder") is not None


has_binder = is_binder_available()
if not has_binder:
    raise RuntimeError(
        "binder is not installed. Please install binder to use this method."
    )
if has_binder:
    from binder.nsql.database import NeuralDB
    from binder.nsql.nsql_exec import Executor
    from binder.utils.normalizer import post_process_sql


class BinderMethod(PromptMethod):
    """TODO: add docstring"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.keys = [os.environ.get("OPENAI_API_KEY")]

    def run(
        self,
        x: Union[str, Dict[str, Union[str, Dict]]],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List[str]]:
        assert isinstance(x, Dict)
        assert isinstance(x["table"], Dict)

        prompt = PromptBuilder.build_prompt(
            x=x,
            in_context_examples=in_context_examples
            if in_context_examples
            else self.kwargs.get("in_context_examples", None),
            prompt_file_path=prompt_file_path
            if prompt_file_path
            else self.kwargs.get("prompt_file_path", None),
            **self.kwargs["generation"]
        )

        nsqls = self.run_lm(prompt, **self.kwargs["generation"])

        # ********* Execution *********
        # TODO: Merge the execution part into the framework.
        # TODO: add a converter of args and kwargs
        executor = Executor(self.kwargs["execution"], keys=self.keys)

        # Execute
        exec_answer_list = []
        for nsql in nsqls:
            try:
                db = NeuralDB(
                    tables=[{"title": x["table"]["page_title"], "table": x["table"]}]
                )
                nsql = post_process_sql(
                    sql_str=nsql,
                    df=db.get_table_df(),
                    process_program_with_fuzzy_match_on_db=self.kwargs["execution"].get(
                        "process_program_with_fuzzy_match_on_db", True
                    ),
                    table_title=x["table"]["page_title"],
                )
                exec_answer = executor.nsql_exec(
                    nsql,
                    db,
                    verbose=kwargs["verbose"]
                    if "verbose" in kwargs
                    else self.kwargs.get("verbose", False),
                )
                exec_answer_list.append(exec_answer)
            except Exception:
                exec_answer = "<error>"
                exec_answer_list.append(exec_answer)

        # ********* Post-hoc *********
        y = HocPoster.post_hoc(
            exec_answer_list,
            aggregation=kwargs["aggregation"]
            if "aggregation" in kwargs
            else self.kwargs["execution"].get("aggregation", None),
        )
        return y
