import json
from typing import Any, Dict, List, Optional, Union

from ...components.post_hoc import HocPoster
from ...components.prompt import PromptBuilder
from ..base_method.method import PromptMethod


class AMAPromptingMethod(PromptMethod):
    """TODO: add docstring"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        with open(self.kwargs["prompt_examples_path"]["questioner_examples"], "r") as f:
            self.questioner_in_context_examples_s = json.load(f)
        with open(
            self.kwargs["prompt_examples_path"]["openended_qa_examples"], "r"
        ) as f:
            self.openended_qa_in_context_examples_s = json.load(f)


    def run(
        self,
        x: Union[str, Dict],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List[str]]:

        assert isinstance(x, Dict)

        y_s = []

        assert len(self.questioner_in_context_examples_s) == len(
            self.openended_qa_in_context_examples_s
        )

        for questioner_in_context_examples, openended_qa_in_context_examples in zip(
            self.questioner_in_context_examples_s,
            self.openended_qa_in_context_examples_s,
        ):
            questioner_prompt = PromptBuilder.build_prompt(
                x=x,
                in_context_examples=questioner_in_context_examples,
                transform=self.kwargs["transform"]["questioner"],
            )

            question = self.run_lm(questioner_prompt, **kwargs)
            assert isinstance(question, str)  # fixme: hard encode
            x["question"] = question.strip()

            openended_qa_prompt = PromptBuilder.build_prompt(
                x=x,
                in_context_examples=openended_qa_in_context_examples,
                transform=self.kwargs["transform"]["openended_qa"],
            )

            answer = self.run_lm(openended_qa_prompt, **kwargs)
            assert isinstance(answer, str)  # fixme: hard encode
            y_s.append(answer.strip())

        y = HocPoster.post_hoc(
            y_s,
            extract=kwargs["extract"]
            if "extract" in kwargs
            else self.kwargs.get("extract", None),
            extraction_regex=kwargs["extraction_regex"]
            if "extraction_regex" in kwargs
            else self.kwargs.get("extraction_regex", "(.*),?.*"),
            aggregation=kwargs["aggregation"]
            if "aggregation" in kwargs
            else self.kwargs.get("aggregation", None),
        )

        return y
