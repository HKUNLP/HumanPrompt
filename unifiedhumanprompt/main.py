import os
from argparse import ArgumentParser

from utils.config_utils import load_config
from tasks.dataset_loader import DatasetLoader
from methods.standard.method import Method


def main():
    config = load_config(args.config)
    dataset = DatasetLoader.load_dataset(**config["dataset"])
    method = Method(**config["method"])

    data_item = dataset['test'][0]
    result = method.run(x=data_item)

    print(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    args = parser.parse_args()

    os.environ['OPENAI_API_KEY'] = "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"

    main()
