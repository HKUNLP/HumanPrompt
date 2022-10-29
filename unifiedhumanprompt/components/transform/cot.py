from base import Transform


class CoTTransform(Transform):

    @staticmethod
    def transform(x, y=None, **kargs):
        """
        Chain of Thought (CoT) is a prompt format a series of intermediate reasoning steps
        which could significantly improves the ability of large language models to perform
        complex reasoning(https://arxiv.org/abs/2201.11903).

        Here is a basic implement of it.

        Args:
            x: input, could be a str or a dict, when it is a str, it is the question itself.
            y:  output, could be a str or a dict, when it is a str, it is the chain-of-thought and answer itself.
            **kargs: other arguments

        Returns: a string of prompt

        """
        if y is None:
            return f"Q: {x['question']}"
        else:
            return f"Q: {x['question']}\nA: {y['chain_of_thought'], y['answer']}"
