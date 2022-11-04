from typing import Any, Callable, Dict, Optional, Union

from manifest import Manifest

from unifiedhumanprompt.components.post_hoc import HocPoster
from unifiedhumanprompt.components.prompt import PromptBuilder


class CoTMethod:
    """Method pipeline class."""

    def __init__(
        self,
        backend: str,
        prompt_file_path: Optional[str] = None,
        transform: Union[Callable, str] = None,
        extraction_words: str = None,
        extract: Union[Callable, str] = None,
        aggregation: Union[Callable, str] = None,
        extraction_regex: str = ".*So the answer is (.*).\n?",
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
        self.extraction_words = extraction_words
        self.extract = extract
        self.aggregation = aggregation
        self.extraction_regex = extraction_regex
        self.kwargs = kwargs

    def run(
        self,
        x: Union[str, Dict],
    ) -> str:
        prompt = PromptBuilder.build_prompt(
            file_path=self.prompt_file_path,
            x=x,
            transform=self.transform,
            extraction_words=self.extraction_words,
        )
        response = self.lm.run(prompt, **self.kwargs)

        y = HocPoster.post_hoc(
            response,
            extract=self.extract,
            aggregation=self.aggregation,
            extraction_regex=self.extraction_regex,
        )
        return y
