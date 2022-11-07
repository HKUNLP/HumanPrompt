from typing import Any, Dict, List, Optional, Union

from ...components.prompt import PromptBuilder
from ..base_method.method import PromptMethod


class ZeroShotCoTMethod(PromptMethod):
    """TODO: add docstring"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def run(
        self,
        x: Union[str, Dict],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        step_1_prompt = PromptBuilder.build_prompt(
            x=x,
            in_context_examples=in_context_examples
            if in_context_examples
            else self.kwargs.get("in_context_examples", None),
            prompt_file_path=prompt_file_path
            if prompt_file_path
            else self.kwargs.get("prompt_file_path", None),
            transform=kwargs["transform"]
            if "transform" in kwargs
            else self.kwargs.get("transform", None),
            extraction_words=kwargs["extraction_words"]
            if "extraction_words" in kwargs
            else self.kwargs.get("extraction_words", None),
        )

        chain_of_thought = self.run_lm(step_1_prompt, **kwargs)
        x["chain_of_thought"] = chain_of_thought

        step_2_prompt = PromptBuilder.build_prompt(
            x=x,
            in_context_examples=in_context_examples
            if in_context_examples
            else self.kwargs.get("in_context_examples", None),
            prompt_file_path=prompt_file_path
            if prompt_file_path
            else self.kwargs.get("prompt_file_path", None),
            transform=kwargs["transform"]
            if "transform" in kwargs
            else self.kwargs.get("transform", None),
            extraction_words=kwargs["extraction_words"]
            if "extraction_words" in kwargs
            else self.kwargs.get("extraction_words", None),
        )

        y = self.run_lm(step_2_prompt, **kwargs)

        return y
