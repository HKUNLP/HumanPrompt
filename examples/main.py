import os

from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    method = AutoMethod.from_config(method_name="binder")

    data = DatasetLoader.load_dataset(dataset_name="wikitq")
    data_item = data["test"][2]

    result = method.run(data_item)
    print(result)
    print(data_item["answer_text"])
