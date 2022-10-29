from typing import Dict, List, Optional, Union, Callable, Any

from manifest import Manifest, Prompt
from unifiedhumanprompt.components.prompt import PromptBuilder


class Method:
    """Method pipeline class."""
    def __init__(
        self,
        backend: str,
        prompt_file_path: Optional[str] = None,
        transform: Union[Callable, str] = None,
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
        self.kwargs = kwargs

    def run(
        self,
        x: Union[str, Dict]
    ) -> str:
        prompt = PromptBuilder.build_prompt(
            file_path=self.prompt_file_path,
            x=x,
            transform=self.transform
        )
        response_text = self.lm.run(prompt, **self.kwargs)
        # TODO: Add extraction if needed
        # TODO: Add aggregation if needed
        y = response_text
        return y
