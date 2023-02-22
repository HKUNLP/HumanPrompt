# HumanPrompt

<p align="middle">
    <a href="https://img.shields.io/badge/version-v0.0.1-blue">
        <img src="https://img.shields.io/badge/version-v0.0.1-blue">
    </a>
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/HKUNLP/HumanPrompt?color=green">
        <img src="https://img.shields.io/github/last-commit/HKUNLP/HumanPrompt?color=green">
    </a>
    <br/>
</p>

HumanPrompt is a framework for easier human-in-the-loop design, manage, sharing, and usage of prompt and prompt methods.

## Content
+ [To start](#to-start)
+ [To accelerate your research](#to-accelerate-your-research)
  - [Config](#config)
  - [Run experiment](#run-experiment)
+ [Architecture](#architecture)
+ [Contributing](#contributing)
  - [Pre-commit](#pre-commit)
+ [Used by](#used-by)
+ [Citation](#citation)


## To start

Firstly, clone this repo, then run:
```bash
pip install -e .
```
This will install humanprompt package and add soft link hub to `./humanprompt/artifacts/hub`.

Then you need to set some environmental variables like OpenAI API key:
```bash
export OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```
Then, it depends on how you will use this repo.
For now, this repo's mission is to help researchers on verifying their ideas. Therefore, we make it really flexible to extend and use.

A minimal example to run a method is as follows:

Our usage is quite simple, it is almost similar if you have used [huggingface transformers](https://huggingface.co/docs/transformers/index) before.

For example, use the Chain-of-Thought on CommonsenseQA:
```python
from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

# Get one built-in method
method = AutoMethod.from_config(method_name="cot")

# Get one dataset, select one example for demo
data = DatasetLoader.load_dataset(dataset_name="commonsense_qa", dataset_split="test")
data_item = data[0]

# Adapt the raw data to the method's input format, (we will improve this part later)
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

# Run the method
result = method.run(data_item)
print(result)
print(data_item)
```

Zero-shot text2SQL:
```python
import os
from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

method = AutoMethod.from_config("db_text2sql")
data = DatasetLoader.load_dataset(dataset_name="spider", dataset_split="validation")
data_item = data[0]

data_item["db"] = os.path.join(
data_item["db_path"], data_item["db_id"], data_item["db_id"] + ".sqlite"
)

result = method.run(data_item)
print(result)
print(data_item)
```

## To accelerate your research
### Config
We adopt "one config, one experiment" paradigm to facilitate research, especially when benchmarking different prompting methods.
In each experiment's config file(.yaml) under `examples/configs/`, you can config the dataset, prompting method, and metrics.

Following is a config file example for Chain-of-Thought method on GSM8K:
```yaml
---
  dataset:
    dataset_name: "gsm8k"                # dataset name, aligned with huggingface dataset if loaded from it
    dataset_split: "test"                # dataset split
    dataset_subset_name: "main"          # dataset subset name, null if not used
    dataset_key_map:                     # mapping original dataset keys to humanprompt task keys to unify the interface
      question: "question"
      answer: "answer"
  method:
    method_name: "cot"                   # method name to initialize the prompting method class
    method_config_file_path: null        # method config file path, null if not used(will be overriden by method_args).
    method_args:
      client_name: "openai"              # LLM API client name, adopted from github.com/HazyResearch/manifest
      transform: "cot.gsm8k.transform_cot_gsm8k.CoTGSM8KTransform"  # user-defined transform class to build the prompts
      extract: "cot.gsm8k.extract_cot_gsm8k.CoTGSM8KExtract"        # user-defined extract class to extract the answers from output
      extraction_regex: ".*The answer is (.*).\n?"                  # user-defined regex to extract the answer from output
      prompt_file_path: "cot/gsm8k/prompt.txt"                      # prompt file path
      max_tokens: 512                    # max generated tokens
      temperature: 0                     # temperature for generated tokens
      engine: code-davinci-002           # LLM engine
      stop_sequence: "\n\n"              # stop sequence for generation
  metrics:
    - "exact_match"                      # metrics to evaluate the results
```
Users can create the `transform` and `extract` classes to customize the prompt generation and answer extraction process.
Prompt file can be replaced or specified according to the user's need.

### Run experiment
To run experiments, you can specify the experiment name and other meta configs in command line under `examples/` directory.

For example, run the following command to run Chain-of-Thought on GSM8K:
```bash
python run_experiment.py
  --exp_name cot-gsm8k
  --num_test_samples 300
```
For new combination of methods and tasks, you can simply add a new config file under `examples/configs/` and run the command.

## Architecture
```text
.
├── examples
│   ├── configs                    # config files for experiments
│   ├── main.py                    # one sample demo script
│   └── run_experiment.py          # experiment script
├── hub                            # hub contains static files for methods and tasks
│   ├── cot                        # method Chain-of-Thought
│   │   ├── gsm8k                  # task GSM8K, containing prompt file and transform/extract classes, etc.
│   │   └── ...
│   ├── ama_prompting              # method Ask Me Anything
│   ├── binder                     # method Binder
│   ├── db_text2sql                # method text2sql
│   ├── react                      # method ReAct
│   ├── standard                   # method standard prompting
│   └── zero_shot_cot              # method zero-shot Chain-of-Thought
├── humanprompt                    # humanprompt package, containing building blocks for the complete prompting pipeline
│   ├── artifacts
│   │   ├── artifact.py
│   │   └── hub
│   ├── components                 # key components for the prompting pipeline
│   │   ├── aggregate              # aggregate classes to aggregate the answers
│   │   ├── extract                # extract classes to extract the answers from output
│   │   ├── post_hoc.py            # post-hoc processing
│   │   ├── prompt.py              # prompt classes to build the prompts
│   │   ├── retrieve               # retrieve classes to retrieve in-context examples
│   │   └── transform              # transform classes to transform the raw data to the method's input format
│   ├── evaluators                 # evaluators
│   │   └── evaluator.py           # evaluator class to evaluate the dataset results
│   ├── methods                    # prompting methods, usually one method is related to one paper
│   │   ├── ama_prompting          # Ask Me Anything(https://arxiv.org/pdf/2210.02441.pdf)
│   │   ├── binder                 # Binder(https://arxiv.org/pdf/2210.02875.pdf)
│   │   └── ...
│   ├── tasks                      # dataset loading and preprocessing
│   │   ├── add_sub.py             # AddSub dataset
│   │   ├── wikitq.py              # WikiTableQuestions dataset
│   │   └── ...
│   ├── third_party                # third party packages
│   └── utils                      # utils
│       ├── config_utils.py
│       └── integrations.py
└── tests                          # test scripts
    ├── conftest.py
    ├── test_datasetloader.py
    └── test_method.py

```

## Contributing
This repository is designed for researchers to give a quick usages and easy manipulation of different prompt methods.
We spent a lot of time on making it easy to extend and use, thus we hope you can contribute to this repo.

If you are interested in contributing your method into this framework, you can:
1. Bring up an issue about your required method, and we will add it into our TODO list and implement as soon as possible.
2. Add your method into `humanprompt/methods` folder yourself. To do that, you should follow the following steps:
   1. Clone the repo.
   2. Create a branch from `main` branch, named you methods.
   3. Commit your code into your branch, you need to:
      1. add code in `./humanprompt/methods`, and add your method into `./humanprompt/methods/your_method_name` folder,
      2. create a hub of your method in `./hub/your_method_name`,
      3. make sure to have an `./examples` folder in `./hub/your_method_name` to config the basic usage this method,
      4. a minimal demo in `./examples` for running and testing your method.
   4. Create a demo of usage in ./examples folder.
   5. Require a PR to merge your branch into `main` branch.
   6. We will handle the last few steps for you to make sure your method is well integrated into this framework.

### Pre-commit
We use pre-commit to control the quality of code.
Before you commit, make sure to run the code below to go over your code and fix the issues.
~~~
pip install pre-commit
pre-commit install # install all hooks
pre-commit run --all-files # trigger all hooks
~~~
You can use `git commit --no-verify` to skip and allow us to handle that later on.

## Used by
- [Batch Prompting](https://github.com/HKUNLP/batch-prompting)

## Citation
If you find this repo useful, please cite our project and [manifest](https://github.com/HazyResearch/manifest):
```bibtex
@software{humanprompt,
  author = {Tianbao Xie and
            Zhoujun Cheng and
            Yiheng Xu and
            Peng Shi and
            Tao Yu},
  title = {A framework for human-readable prompt-based method with large language models},
  howpublished = {\url{https://github.com/hkunlp/humanprompt}},
  year = 2022,
  month = October
}
```

```bibtex
@misc{orr2022manifest,
  author = {Orr, Laurel},
  title = {Manifest},
  year = {2022},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/HazyResearch/manifest}},
}
```
