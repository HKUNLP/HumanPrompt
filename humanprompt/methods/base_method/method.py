import abc
import inspect
from typing import Any, Dict, List, Optional, Union

import backoff
from manifest import Manifest

MAX_TRIES = 3


class PromptMethod(abc.ABC):
    def __init__(self, **kwargs: Any):
        """
        Receives kwargs and prepare a manifest connection for prompt method running.
        Args:
            **kwargs:
        """
        self.kwargs = kwargs

        init_params = {}
        for param in kwargs:
            if param in inspect.getfullargspec(Manifest.__init__).args:
                init_params[param] = kwargs[param]
        self.manifest = Manifest(**init_params)

    @backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)
    def run_lm(self, prompt: str, **kwargs: Any) -> Union[str, List[str]]:
        """
        Run the language model with the given prompt.
        Only the acceptable kwargs are passed to the language model.
        Args:
            prompt: prompt to run the language model with
            **kwargs:

        Returns: The response from the language model
        """
        run_required_params = inspect.getfullargspec(self.manifest.run).args
        client_request_required_params = self.manifest.client.get_model_inputs()

        run_params = {}

        # Default from self kwargs
        for param in self.kwargs:
            if param in run_required_params + client_request_required_params:
                run_params[param] = self.kwargs[param]

        # Override from kwargs
        for param in kwargs:
            if param in run_required_params + client_request_required_params:
                run_params[param] = kwargs[param]

        return self.manifest.run(prompt, overwrite_cache=True, **run_params)

    @abc.abstractmethod
    def run(
        self,
        x: Union[str, Dict],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List[str]]:
        """
        Run the method with the given x and optional in_context_examples or prompt_file_path.
        Args:
            x: The input to the method
            in_context_examples: In context examples for in-context learning
            prompt_file_path: Prompt file for prompting the language model
            **kwargs:

        Returns: The result of this method

        """
        pass
