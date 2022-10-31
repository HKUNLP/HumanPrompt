from typing import Dict, List, Optional, Union, Callable, Any
from manifest import Manifest
from unifiedhumanprompt.components.prompt import PromptBuilder
from unifiedhumanprompt.components.post_hoc import HocPoster


class Method:
    """Method pipeline class."""

    def __init__(
            self,
            backend: str,
            prompt_file_path: Optional[str] = None,
            transform: Union[Callable, str] = None,
            extract: Union[Callable, str] = None,
            **kwargs: Any
    ):
        self.lm = Manifest(
            client_name=backend,
            client_connection=None,
            cache_name="noop",
            cache_connection=None,
            session_id=None,
        )
        self.prompt_file_path = prompt_file_path
        self.transform = transform
        self.extract = extract
        self.kwargs = kwargs

    def run(
            self,
            x: Union[str, Dict],
            extraction_regex: str = ".*So the answer is (.*).\n?"
    ) -> str:
        prompt = PromptBuilder.build_prompt(
            file_path=self.prompt_file_path,
            x=x,
            transform=self.transform
        )
        response = self.lm.run(prompt, **self.kwargs)

        if isinstance(response, str):
            y = HocPoster.post_hoc(
                response,
                extract=self.extract,
                extraction_regex=extraction_regex
            )
            return y
        elif isinstance(response, list):
            y_s = []
            for response_text in response:
                y_s.append(
                    HocPoster.post_hoc(
                        response_text,
                        extract=self.extract,
                        extraction_regex=extraction_regex
                    )
                )



            return y
