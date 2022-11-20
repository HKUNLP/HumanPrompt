# HumanPrompt


## Description
HumanPrompt is a framework for easier human-in-the-loop design, manage, sharing, and usage of prompt and prompt methods.

## To run
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
```python
from humanprompt.methods.auto.method_auto import AutoMethod
from humanprompt.tasks.dataset_loader import DatasetLoader

# Get one built-in method
method = AutoMethod.from_config(method_name="cot")

# Get one dataset, select one example for demo
data = DatasetLoader.load_dataset(dataset_name="commonsense_qa")
data_item = data["test"][0]

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
## Project Architecture
to be added

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

### Things to be done in the future

#### Before 2022.12
- [ ] Add parameter hint for methods
- [ ] Add verbose in each method
- [ ] Add README and credict to each method
- [ ] Add methods(ReAct, self-ask) that requires external API
- [ ] Add one program generation method(prompt prog)
- [ ] Add one data generation method(ZeroGenI)
- [ ] Add one method on dialogue component task(IC-DST)
- [ ] Add one text-to-sql task(DB SP)
- [ ] Add one more aggregation method for code generation(MBR-EXEC)
- [ ] Add one more aggregation method used in ama prompting(METAL-AMA)
- [ ] Support batch running function
- [ ] Start UI construction
- [x] Add AMA_prompting
- [x] Shorten the repo name to be published on PyPI
- [x] Add more tests
- [x] Add one skg method(Binder)
- [x] Init the experiments part
- [x] Add CLI
- [x] Add static checkings
- [x] Derive the manifest from the method, make full use of its init, run and run_batch(In progress)
- [x] Support in-context examples when using method.run
- [x] Path and preparation for PyPI
- [x] Add AutoMethod Factory

#### Before 2023.1
- [ ] PyPI publish
- [ ] Demo website
- [ ] twitter
