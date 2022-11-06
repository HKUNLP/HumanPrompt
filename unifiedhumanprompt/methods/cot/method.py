from typing import Any, Dict, Union, List
from ...components.post_hoc import HocPoster
from ...components.prompt import PromptBuilder
from ..base_method.method import PromptMethod


class CoTMethod(PromptMethod):
    """TODO: add docstring"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def run(
            self,
            x: Union[str, Dict],
            in_context_examples: List[Dict] = None,
            prompt_file_path=None,
            **kwargs: Any
    ) -> str:

        prompt = PromptBuilder.build_prompt(
            x=x,
            in_context_examples=in_context_examples if in_context_examples else self.kwargs.get('in_context_examples', None),
            prompt_file_path=prompt_file_path if prompt_file_path else self.kwargs.get('prompt_file_path', None),
            transform=kwargs['transform'] if 'transform' in kwargs else self.kwargs.get('transform', None),
            extraction_words=kwargs['extraction_words'] if 'extraction_words' in kwargs else self.kwargs.get('extraction_words', None)
        )

        response = self.run_lm(prompt, **kwargs)

        y = HocPoster.post_hoc(
            response,
            extract=kwargs['extract'] if 'extract' in kwargs else self.kwargs.get('extract', None),
            aggregation=kwargs['aggregation'] if 'aggregation' in kwargs else self.kwargs.get('aggregation', None),
            extraction_regex=kwargs['extraction_regex'] if 'extraction_regex' in kwargs else self.kwargs.get('extraction_regex', ".*So the answer is (.*).\n?"),
        )
        return y
