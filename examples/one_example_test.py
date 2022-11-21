import os

from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    method = AutoMethod.from_config("db_text2sql")

    dataset = DatasetLoader.load_dataset(
        dataset_name="spider", dataset_split="validation"
    )
    data_item = dataset[0]  # type: ignore
    data_item["db"] = os.path.join(
        data_item["db_path"], data_item["db_id"], data_item["db_id"] + ".sqlite"
    )

    result = method.run(data_item)
    print(result)
