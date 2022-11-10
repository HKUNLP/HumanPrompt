import os

from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    method = AutoMethod.from_config(method_name="binder")

    dataset = DatasetLoader.load_dataset(dataset_name="wikitq")
    data_item = dataset["test"][0]
    result = method.run(data_item)
    print(result)
