import os

from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    method = AutoMethod.from_config("ama_prompting")

    dataset = DatasetLoader.load_dataset(dataset_name="super_glue", name="cb")
    data_item = dataset["test"][0]  # type: ignore

    # Adapt the raw data to the format of the method
    data_item["passage"] = data_item["premise"]
    data_item["statement"] = data_item["hypothesis"]

    result = method.run(data_item)
    print(result)
