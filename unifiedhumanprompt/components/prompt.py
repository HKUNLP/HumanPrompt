from typing import Any, Callable, Dict, List, Optional, Union

from .transform.transform_factory import TransformFactory


class PromptBuilder:
    """Prompt builder tool class."""

    @staticmethod
    def build_prompt(
        x: Union[str, Dict],
        in_context_examples: List[Dict] = None,
        prompt_file_path: Optional[str] = None,
        transform: Union[str, Callable] = None,
        **kwargs: Any
    ) -> str:
        """
        Build prompt from x, in_context_examples, and prompt_file_path.

        Args:
            x: Input to the prompt.
            in_context_examples: List of examples to be used in context.
            prompt_file_path: Path to file containing prompt.
            transform: Transform to apply to x.
            **kwargs: Additional arguments to pass to the transform.

        Returns:
            Prompt string.
        """

        # First, if the file path is specified, load the prompt from the file.
        if prompt_file_path:
            return PromptBuilder.build_prompt_from_file(
                prompt_file_path, x, transform, **kwargs
            )

        # If the file path is not specified, check if there are in-context examples.
        elif in_context_examples:
            # If there are in-context examples, add them to the prompt.
            return PromptBuilder.build_prompt_from_examples(
                x, in_context_examples, transform, **kwargs
            )

        elif x:
            # If there are no in-context examples, just use the input.
            return PromptBuilder.build_one_prompt(x, transform, **kwargs)

        else:
            raise ValueError("No prompt pattern for this set of args.")

    @staticmethod
    def build_one_prompt(
        x: Union[str, Dict],
        transform: Union[str, Callable],
        y: Union[str, Dict] = None,
        **kwargs: Any
    ) -> str:

        if isinstance(transform, Callable):
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
        **kwargs: Any
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
        **kwargs: Any
    ) -> str:
        # todo: add spec for x, in_context_examples
        in_context_examples_prompt = ""
        for in_context_example in in_context_examples:
            in_context_examples_prompt += (
                PromptBuilder.build_one_prompt(
                    x=in_context_example["x"],
                    y=in_context_example["y"],
                    transform=transform,
                    **kwargs
                )
                + "\n\n"
            )

        x_prompt = PromptBuilder.build_one_prompt(x=x, transform=transform, **kwargs)

        prompt = in_context_examples_prompt + x_prompt
        return prompt
