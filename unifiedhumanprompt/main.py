from argparse import ArgumentParser
import wandb
import json

from utils.config_utils import load_config


def main():
    config = load_config(args.config)
    print(json.dumps(config, indent=2))
    # TODO: Add wandb
    # wandb.init(project='unifiedhumanprompt', config=config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    args = parser.parse_args()

    main()
