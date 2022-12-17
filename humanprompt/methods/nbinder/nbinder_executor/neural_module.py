from typing import Any, Dict, List, Optional, Union

from ....components.post_hoc import HocPoster
from ....components.prompt import PromptBuilder
from ....methods.base_method.method import PromptMethod


class NeuralModule(PromptMethod):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def run(
        self,
        x: Union[str, Dict[str, Union[str, Dict]]],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List[str]]:
        # Add retrieve method if needed.
        prompt = PromptBuilder.build_prompt(x=x, **self.kwargs["execution"])
        response = self.run_lm(prompt, **self.kwargs["execution"])

        if self.kwargs["execution"]["temperature"] == 0:
            return response
        else:
            y = HocPoster.post_hoc(
                response,
                extract=self.kwargs["execution"]["extract"],
                extraction_regex=self.kwargs["execution"]["extraction_regex"],
                aggregation=self.kwargs["execution"]["aggregation"],
            )
            return y
