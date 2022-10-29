from typing import Callable, Union, List
from manifest import Prompt


class PromptBuilder:
    """Prompt builder tool class."""

    @staticmethod
    def build_prompt(
            file_path: str = None,
            input_x: Union[str, List[str]] = None,
            output_y: Union[str, List[str]] = None,
            transform: Callable = None,
            exemplar_splitter: str = '\n',
    ) -> Prompt:
        # TODO: Support combination of few-shot prompt and test prompt
        few_shot_prompt = ""
        if file_path is not None:
            # Load from file
            with open(file_path, 'r') as f:
                few_shot_prompt = f.read()

        if isinstance(input_x, str):
            input_x = [input_x]
        if isinstance(output_y, str):
            output_y = [output_y]
        if input_x and output_y:
            # Transform to the few-shot prompt
            prompt_list = []
            for x, y in zip(input_x, output_y):
                prompt_list.append(transform(x, y))
            prompt = exemplar_splitter.join(prompt_list)
        elif input_x:
            # Transform to the test prompt
            assert len(input_x) == 1, "Only one input is allowed for test prompt."
            test_prompt = transform(input_x[0])
            if few_shot_prompt:
                prompt = exemplar_splitter.join([few_shot_prompt, test_prompt])
            else:
                prompt = test_prompt
        else:
            raise ValueError(
                "Please provide either a prompt string or a file path or both input and output."
            )

        return Prompt(prompt)