from typing import Any, Dict, List, Optional, Union
import os

from ...components.post_hoc import HocPoster
from ...components.prompt import PromptBuilder
from ...methods.base_method.method import PromptMethod
from .nsql.nsql_exec import Executor, NeuralDB
from .generation.generator import Generator

from .utils.normalizer import post_process_sql

# todo: Used the majority of the code from aggregator.
from .utils.utils import majority_vote

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


class BinderMethod(PromptMethod):
    """TODO: add docstring"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.keys = [os.environ.get("OPENAI_API_KEY")]

    def run(
            self,
            x: Union[str, Dict],
            in_context_examples: List[Dict] = None,
            prompt_file_path: Optional[str] = None,
            **kwargs: Any
    ) -> Union[str, List[str]]:
        args = None
        # ********* Generation *********
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))

        db = NeuralDB(
            tables=[{'title': x['table']['page_title'], 'table': x['table']}]
        )
        x['table'] = db.get_table_df()
        x['title'] = db.get_table_title()

        # todo: pack into the transform methods.
        generator = Generator(args, self.keys)
        n_shots = kwargs["n_shots"] if "n_shots" in kwargs else self.kwargs.get("n_shots", 14)
        few_shot_prompt = generator.build_few_shot_prompt_from_file(
            file_path=prompt_file_path if prompt_file_path else self.kwargs.get("prompt_file_path", None),
            n_shots=n_shots
        )
        generate_prompt = generator.build_generate_prompt(
            data_item=x,
            generate_type=('nsql',)
        )
        prompt = few_shot_prompt + "\n\n" + generate_prompt

        # Ensure the input length fit Codex max input tokens by shrinking the n_shots
        while len(tokenizer.tokenize(prompt)) >= self.kwargs['max_prompt_tokens']:
            n_shots -= 1
            assert n_shots >= 0
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=prompt_file_path if prompt_file_path else self.kwargs.get("prompt_file_path", None),
                n_shots=n_shots
            )
            prompt = few_shot_prompt + "\n\n" + generate_prompt

        # todo: pack into the transform methods.

        # prompt = PromptBuilder.build_prompt(
        #     x=x,
        #     in_context_examples=in_context_examples
        #     if in_context_examples
        #     else self.kwargs.get("in_context_examples", None),
        #     prompt_file_path=prompt_file_path
        #     if prompt_file_path
        #     else self.kwargs.get("prompt_file_path", None),
        #     transform=kwargs["transform"]
        #     if "transform" in kwargs
        #     else self.kwargs.get("transform", None),
        #     extraction_words=kwargs["extraction_words"]
        #     if "extraction_words" in kwargs
        #     else self.kwargs.get("extraction_words", None),
        # )

        nsqls = self.run_lm(prompt, **kwargs)

        # ********* Execution *********
        table = x['table']
        title = table['page_title']

        executor = Executor(None)
        # Execute
        exec_answer_list = []
        nsql_exec_answer_dict = dict()
        for idx, nsql in nsqls:
            try:
                db = NeuralDB(
                    tables=[{"title": title, "table": table}]
                )
                nsql = post_process_sql(
                    sql_str=nsql,
                    df=db.get_table_df(),
                    process_program_with_fuzzy_match_on_db=args.process_program_with_fuzzy_match_on_db,
                    table_title=title
                )
                exec_answer = executor.nsql_exec(nsql, db, verbose=args.verbose)
                if isinstance(exec_answer, str):
                    exec_answer = [exec_answer]
                nsql_exec_answer_dict[nsql] = exec_answer
                exec_answer_list.append(exec_answer)
            except Exception as e:
                exec_answer = '<error>'
                exec_answer_list.append(exec_answer)

        # ********* Post-hoc *********

        # Majority vote to determine the final prediction answer
        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=nsqls,
            pred_answer_list=exec_answer_list,
            allow_none_and_empty_answer=args.allow_none_and_empty_answer,
            answer_placeholder=args.answer_placeholder,
            vote_method=args.vote_method,
            answer_biased=args.answer_biased,
            answer_biased_weight=args.answer_biased_weight
        )


        return pred_answer
