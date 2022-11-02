from typing import Dict, List, Optional, Union, Callable, Any
from manifest import Manifest
from unifiedhumanprompt.components.prompt import PromptBuilder
from unifiedhumanprompt.components.post_hoc import HocPoster


class Method():
    """Method pipeline class."""

    def __init__(
            self,
            backend: str,
            # todo: too wide
            transform: Union[Callable, str] = None,
            extraction_words: str = None,
            **kwargs: Any
    ):
        # todo: derive it to become a class?
        self.lm = Manifest(
            client_name=backend,
            client_connection=None,
            cache_name="noop",
            cache_connection=None,
            session_id=None,
        )
        self.transform = transform
        self.extraction_words = extraction_words
        self.kwargs = kwargs

    def run(
            self,
            x: Union[str, Dict],
    ) -> str:
        step_1_prompt = PromptBuilder.build_prompt(
            x=x,
            transform=self.transform,
            extraction_words=self.extraction_words
        )

        # todo: why we assume kwargs is always for the lm?
        chain_of_thought = self.lm.run(step_1_prompt, **self.kwargs)

        x['chain_of_thought'] = chain_of_thought

        step_2_prompt = PromptBuilder.build_prompt(
            x=x,
            transform=self.transform,
            extraction_words=self.extraction_words
        )

        y = self.lm.run(step_2_prompt, **self.kwargs)

        return y
