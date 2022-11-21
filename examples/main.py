import os

from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    method = AutoMethod.from_config(method_name="cot")

    dataset = DatasetLoader.load_dataset(
        dataset_name="commonsense_qa",
        dataset_split="validation",
        # TODO: Maybe add default dataset_key_map for each dataset
        dataset_key_map={
            "question": "question",
            "choices": "choices",
            "answer": "answerKey",
            "id": "id"
        }
    )
    data_item = dataset[0]

    print(data_item)

    result = method.run(data_item)
    print(result)
    print(data_item["answer"])
