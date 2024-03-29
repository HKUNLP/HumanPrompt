from typing import Any, Dict, List, Optional, Union

from ...components.prompt import PromptBuilder
from ..base_method.method import PromptMethod


class StandardMethod(PromptMethod):
    """TODO: add docstring"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def run(
        self,
        x: Union[str, Dict],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List[str]]:
        prompt = PromptBuilder.build_prompt(
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
        )

        response = self.run_lm(prompt, **kwargs)
        return response
