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
"""StrategyQA Dataset"""
# mypy: ignore-errors
import json
from typing import Dict, Iterator, List, Tuple

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{geva2021strategyqa,
  title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
  author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
  journal = {Transactions of the Association for Computational Linguistics (TACL)},
  year = {2021},
}
"""
_DESCRIPTION = """\
StrategyQA is a question-answering benchmark focusing on open-domain questions where the required reasoning steps are implicit in the question and should be inferred using a strategy. StrategyQA includes 2,780 examples, each consisting of a strategy question, its decomposition, and evidence paragraphs.
"""
_HOMEPAGE = "https://allenai.org/data/strategyqa"
_LICENSE = "MIT License"
_URL = "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip"


class StrategyQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="strategyqa",
            version=VERSION,
            description="StrategyQA Dataset",
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
                "answer": datasets.Value("string"),
                "facts": datasets.features.Sequence(datasets.Value("string")),
                "decomposition": datasets.features.Sequence(datasets.Value("string")),
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
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/strategyqa_train.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/strategyqa_test.json",
                },
            ),
        ]

    def _generate_examples(
        self, data_filepath: str
    ) -> Iterator[Tuple[int, Dict[str, dict]]]:
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, ex in enumerate(data):
                if ex.get("answer", None) is not None:
                    # Train set
                    yield idx, {
                        "id": ex["qid"],
                        "question": ex["question"],
                        "answer": ex["answer"],
                        "facts": ex["facts"],
                        "decomposition": ex["decomposition"],
                    }
                else:
                    # Test set
                    # TODO: Test set answer is not publicly available.
                    yield idx, {
                        "id": ex["qid"],
                        "question": ex["question"],
                        "answer": None,
                        "facts": None,
                        "decomposition": None,
                    }
