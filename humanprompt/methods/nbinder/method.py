import copy
import re
from typing import Any, Dict, List, Optional, Union

from ...components.post_hoc import HocPoster
from ...components.prompt import PromptBuilder
from ...methods.base_method.method import PromptMethod
from .nbinder_executor.execute import Executor


class NBinderMethod(PromptMethod):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.N = int(
            self.kwargs["generation"]["n"] / 5
        )  # todo: add a more accurate calculator
        self.kwargs["generation"]["n"] = int(self.kwargs["generation"]["n"] / self.N)
        self.kwargs["generation"]["top_k_return"] = int(
            self.kwargs["generation"]["top_k_return"] / self.N
        )

    def run(
        self,
        x: Union[str, Dict[str, Union[str, Dict]]],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List]:
        assert isinstance(x, Dict)
        verbose = self.kwargs.get("verbose", False)

        # ********* Annotation *********
        prompt = PromptBuilder.build_prompt(x=x, **self.kwargs["generation"])

        binder_programs: List[str] = []
        # when the n_sampling is greater than 10, it will be roughly the same as the original n_sampling
        for i in range(self.N):
            binder_programs.extend(self.run_lm(prompt, **self.kwargs["generation"]))

        # ********* Execution *********
        executor = Executor(**self.kwargs)
        exec_answer_list = []
        pure_programs = []
        sub_qas_s = []

        for binder_program in binder_programs:
            try:
                success, exec_message, pure_program, sub_qas = executor.execute(
                    binder_program, x, verbose
                )
                if success:
                    exec_answer_list.append(exec_message)
                    pure_programs.append(pure_program)
                    sub_qas_s.append(sub_qas)
                else:
                    fix_times = self.kwargs["execution"]["fix_times"]
                    assert fix_times > 0
                    while fix_times > 0:
                        # ********* Edit and fix *********
                        if verbose:
                            print(exec_message)
                            print("Retrying fix it..., {} times left".format(fix_times))

                        line = int(
                            re.search(r"File \S+, line (\d+),", exec_message).group(1)
                        )
                        fix_program_prompt = PromptBuilder.build_prompt(
                            x=x, **self.kwargs["generation"]
                        )
                        fix_program_prompt += "\n".join(
                            pure_program.split("\n")[: line - 1]
                        )
                        new_args = copy.deepcopy(self.kwargs["generation"])
                        new_args["n"], new_args["top_k_return"] = (
                            1,
                            1,
                        )  # fixme: Add to hyper-parameters
                        re_generated = self.run_lm(fix_program_prompt, **new_args)
                        assert isinstance(re_generated, str), "re_generated is not str"
                        fix_binder_program = (
                            "\n".join(pure_program.split("\n")[: line - 1])
                            + "\n"
                            + re_generated
                        )
                        success, exec_message, pure_program, sub_qas = executor.execute(
                            fix_binder_program, x, verbose
                        )

                        if success:
                            exec_answer_list.append(exec_message)
                            pure_programs.append(pure_program)
                            sub_qas_s.append(sub_qas)
                            break
                        else:
                            fix_times -= 1
                            if fix_times == 0:
                                if verbose:
                                    print("Failed to fix it.")
                                exec_answer_list.append("<error>")
                                pure_programs.append(pure_program)
                                sub_qas_s.append(sub_qas)
            except Exception as e:
                if verbose:
                    print("Execution error,", e)
                exec_answer_list.append("<error>")
                pure_programs.append(None)
                sub_qas_s.append([])

        # ********* Post-hoc *********
        y = HocPoster.post_hoc(
            exec_answer_list,
            aggregation=self.kwargs["aggregation"],
        )
        return [y, binder_programs, pure_programs, exec_answer_list, sub_qas_s]
