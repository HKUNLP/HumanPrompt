from typing import Any, Dict, Union

from humanprompt.components.transform.transform_base import Transform


class ZeroShotCoTTransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Zero-shot Chain of Thought (CoT) is a prompt format use the same single prompt template,
        significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks
        (https://arxiv.org/abs/2205.11916)

        """
        if isinstance(x, str) or isinstance(y, str):
            raise NotImplementedError

        transformed = f"Q: {x['question']}\n"
        if "context" in x:
            transformed += f"{x['context']}\n"
        transformed += "A: "

        if "extraction_words" in kwargs:
            extraction_words = kwargs["extraction_words"]
        else:
            extraction_words = "The answer is"

        if y:
            transformed += f"{y['chain_of_thought']} {extraction_words} {y['answer']}"

        else:
            # cot trigger(let's think step by step) for step 1 and step 2.
            if "cot_trigger" in kwargs:
                cot_trigger = f"{kwargs['cot_trigger']} "
            else:
                cot_trigger = "Let's think step by step. "

            if "chain_of_thought" not in x.keys():
                # 【1st prompt】
                # Reasoning Extraction
                transformed += f"{cot_trigger}"
            else:
                # 【2nd prompt】
                # Answer Extraction
                chain_of_thought = x["chain_of_thought"]
                transformed += f"{cot_trigger} {chain_of_thought} {extraction_words}"

        return transformed
