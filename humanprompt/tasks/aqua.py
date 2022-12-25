# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AQuA Dataset"""
# mypy: ignore-errors
from typing import Any, Dict, Iterator, List, Tuple
import json

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{ling-etal-2017-program,
    title = "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems",
    author = "Ling, Wang  and
      Yogatama, Dani  and
      Dyer, Chris  and
      Blunsom, Phil",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1015",
    doi = "10.18653/v1/P17-1015",
    pages = "158--167",
    abstract = "Solving algebraic word problems requires executing a series of arithmetic operations{---}a program{---}to obtain a final answer. However, since programs can be arbitrarily complicated, inducing them directly from question-answer pairs is a formidable challenge. To make this task more feasible, we solve these problems by generating answer rationales, sequences of natural language and human-readable mathematical expressions that derive the final answer through a series of small steps. Although rationales do not explicitly specify programs, they provide a scaffolding for their structure via intermediate milestones. To evaluate our approach, we have created a new 100,000-sample dataset of questions, answers and rationales. Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.",
}
"""
_DESCRIPTION = """
AQuA dataset.
"""
_HOMEPAGE = "https://github.com/deepmind/AQuA"
_LICENSE = "MIT License"
_URL = (
    "https://raw.githubusercontent.com/deepmind/AQuA/master/"
)
_TRAINING_FILE = "train.json"
_DEV_FILE = "dev.json"
_TEST_FILE = "test.json"
_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    "test": f"{_URL}{_TEST_FILE}"
}


class SVAMP(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="aqua",
            version=VERSION,
            description="AQuA Dataset",
        ),
    ]

    def __init__(
        self, *args: List, writer_batch_size: int = None, **kwargs: Dict
    ) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache: Dict = dict()

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "choices": datasets.features.Sequence(
                    {
                        "label": datasets.Value("string"),
                        "text": datasets.Value("string"),
                    }
                ),
                "answer": datasets.Value("string"),
                "rationale": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            )
        ]

    def _generate_examples(self, filepath: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
                data = f.readlines()
                for idx, line in enumerate(data):
                    ex = json.loads(line)
                    choices = {
                        "label": ['A', 'B', 'C', 'D', 'E'],
                        "text": [choice[choice.find(")")+1:] for choice in ex["options"]]
                    }
                    yield idx, {
                        "id": str(idx),
                        "question": ex["question"],
                        "choices": choices,
                        "answer": ex["correct"],
                        "rationale": ex["rationale"]
                    }
