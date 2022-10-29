import os
from argparse import ArgumentParser
from utils.config_utils import load_config
from methods.base import Method


class Dataset:
    own_dataset = {
        "default": "11"
    }

    @staticmethod
    def load_dataset():
        print(Dataset.own_dataset)


def main():
    # TODO: Use config
    config = load_config(args.config)
    # TODO: Use dataset
    dataset = Dataset.load_dataset()
    # TODO: Add wandb
    # wandb.init(project='unifiedhumanprompt', config=config)

    os.environ['OPENAI_API_KEY'] = "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"
    method = Method(
        backend="openai",
        transform='default',
    )
    print(method.run(x='Is the grass red?'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    args = parser.parse_args()

    main()
