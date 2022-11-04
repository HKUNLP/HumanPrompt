import os

from unifiedhumanprompt.methods.auto.method_auto import AutoMethod
from unifiedhumanprompt.tasks.dataset_loader import DatasetLoader

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    method = AutoMethod.from_config(method_name="cot")

    dataset = DatasetLoader.load_dataset(dataset_name="commonsense_qa")
    data_item = dataset["test"][0]
    data_item["context"] = "Answer choices: {}".format(
        " ".join(
            [
                "({}) {}".format(label.lower(), text.lower())
                for label, text in zip(
                    data_item["choices"]["label"], data_item["choices"]["text"]
                )
            ]
        )
    )
    result = method.run(data_item)
    print(result)
