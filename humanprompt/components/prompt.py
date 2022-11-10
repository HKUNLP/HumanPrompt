from typing import Any, Callable, Dict, List, Optional, Union

from .transform.transform_factory import TransformFactory
from transformers import AutoTokenizer


class PromptBuilder:
    """Prompt builder tool class."""

    @staticmethod
    def build_prompt(
            x: Union[str, Dict],
            description: str = None,
            in_context_examples: List[Dict] = None,
            prompt_file_path: Optional[str] = None,
            transform: Union[str, Callable] = None,
            n_shots: int = None,
            do_trim: bool = False,
            max_tokens: int = 8001,
            **kwargs: Any,
    ) -> str:
        """
        Build prompt from x, in_context_examples, and prompt_file_path.

        Args:

            x: Input to the prompt.
            description: Description to be added at the start of the prompt.
            in_context_examples: List of examples to be used in context.
            prompt_file_path: Path to file containing prompt.
            transform: Transform to apply to x.
            n_shots: Number of shots to use in the prompt.
            do_trim: Whether to trim the prompt to max_tokens.
            max_tokens: Maximum number of tokens to use in the prompt.
            **kwargs: Additional arguments to pass to the transform.

        Returns:
            Prompt string.
        """

        # First, if the file path is specified, load the prompt from the file.
        if prompt_file_path:
            prompt_body = PromptBuilder.build_prompt_from_file(
                prompt_file_path, x, transform, **kwargs
            )

        # If the file path is not specified, check if there are in-context examples.
        elif in_context_examples:
            # If there are in-context examples, add them to the prompt.
            prompt_body = PromptBuilder.build_prompt_from_examples(
                x, in_context_examples, transform, n_shots, **kwargs
            )

        elif x:
            # If there are no in-context examples, just use the input.
            prompt_body = PromptBuilder.build_one_prompt(x, transform, **kwargs)

        else:
            raise ValueError("No prompt pattern for this set of args.")

        # If the description is specified, add it to the prompt.
        if description:
            prompt = f"{description}\n\n{prompt_body}"
        else:
            prompt = prompt_body

        # if trim is specified, trim the prompt.
        if do_trim:
            prompt = PromptBuilder.trim_prompt(prompt, max_tokens)

        return prompt

    @staticmethod
    def build_one_prompt(
            x: Union[str, Dict],
            transform: Union[str, Callable],
            y: Union[str, Dict] = None,
            **kwargs: Any,
    ) -> str:

        if callable(transform):
            if x and y:
                return transform(x, y, **kwargs)
            elif x:
                return transform(x, **kwargs)
            else:
                raise ValueError("x is required for transform")

        elif isinstance(transform, str):
            if x and y:
                return TransformFactory.get_transform(transform).transform(
                    x, y, **kwargs
                )
            elif x:
                return TransformFactory.get_transform(transform).transform(x, **kwargs)
            else:
                raise ValueError("x is required for transform")

        else:
            raise ValueError("transform must be a callable or a string")

    @staticmethod
    def build_prompt_from_file(
            prompt_file_path: str,
            x: Union[str, Dict] = None,
            transform: Union[str, Callable] = None,
            **kwargs: Any,
    ) -> str:
        prompt = ""

        if prompt_file_path and not x:
            with open(prompt_file_path, "r") as f:
                prompt += f.read()
        elif prompt_file_path and x:
            with open(prompt_file_path, "r") as f:
                prompt += f.read().rstrip()
                prompt += "\n\n"
                prompt += PromptBuilder.build_one_prompt(x, transform, **kwargs)

        return prompt

    @staticmethod
    def build_prompt_from_examples(
            x: Union[str, Dict],
            in_context_examples: List[Dict],
            transform: Union[str, Callable],
            n_shots: int = None,
            **kwargs: Any,
    ) -> str:
        # todo: add spec for x, in_context_examples
        in_context_examples_prompt = ""
        n_shots = len(in_context_examples) if n_shots is None else n_shots
        for in_context_example in in_context_examples[:n_shots]:
            in_context_examples_prompt += (
                    PromptBuilder.build_one_prompt(
                        x=in_context_example["x"],
                        y=in_context_example["y"],
                        transform=transform,
                        **kwargs,
                    )
                    + "\n\n"
            )

        x_prompt = PromptBuilder.build_one_prompt(x=x, transform=transform, **kwargs)

        prompt = in_context_examples_prompt + x_prompt
        return prompt

    @staticmethod
    def parse_prompt_to_examples(
            prompt: str
    ) -> List[str]:
        """

        Args:
            prompt: whole prompt string

        Returns:

        """
        # todo: change the logic
        return prompt.split("\n\n")


    @staticmethod
    def trim_examples(prompt: str, max_tokens: int) -> str:
        """Trim prompt to fit into the max tokens."""
        items = PromptBuilder.parse_prompt_to_examples(prompt)
        few_shot_prompt, generate_prompt = items[:-1], items[-1]
        tokenizer = AutoTokenizer.from_pretrained("gpt2") # fixme: hardcode
        n_shots = len(few_shot_prompt)

        # Ensure the input length fit Codex max input tokens by shrinking the n_shots
        while len(tokenizer.tokenize(prompt)) >= max_tokens:
            n_shots -= 1
            assert n_shots >= 0, "Too long, even unable to do zero-shot!"
            prompt = "\n\n".join(few_shot_prompt) + "\n\n" + generate_prompt

        return prompt
