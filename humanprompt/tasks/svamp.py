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
"""SVAMP Dataset"""
# mypy: ignore-errors
import csv
from typing import Any, Dict, Iterator, List, Tuple
import nltk

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{patel-etal-2021-nlp,
    title = "Are {NLP} Models really able to Solve Simple Math Word Problems?",
    author = "Patel, Arkil  and
      Bhattamishra, Satwik  and
      Goyal, Navin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.168",
    doi = "10.18653/v1/2021.naacl-main.168",
    pages = "2080--2094",
    abstract = "The problem of designing NLP solvers for math word problems (MWP) has seen sustained research activity and steady gains in the test accuracy. Since existing solvers achieve high performance on the benchmark datasets for elementary level MWPs containing one-unknown arithmetic word problems, such problems are often considered {``}solved{''} with the bulk of research attention moving to more complex MWPs. In this paper, we restrict our attention to English MWPs taught in grades four and lower. We provide strong evidence that the existing MWP solvers rely on shallow heuristics to achieve high performance on the benchmark datasets. To this end, we show that MWP solvers that do not have access to the question asked in the MWP can still solve a large fraction of MWPs. Similarly, models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy. Further, we introduce a challenge dataset, SVAMP, created by applying carefully chosen variations over examples sampled from existing datasets. The best accuracy achieved by state-of-the-art models is substantially lower on SVAMP, thus showing that much remains to be done even for the simplest of the MWPs.",
}
"""
_DESCRIPTION = """
SVAMP is a challenge set of ASDiv-A and MAWPS.
"""
_HOMEPAGE = "https://github.com/arkilpatel/SVAMP"
_LICENSE = "MIT License"
_URL = (
    "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/mawps-asdiv-a_svamp/"
)
_TRAINING_FILE = "train.csv"
_DEV_FILE = "dev.csv"
_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
}


class SVAMP(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="svamp",
            version=VERSION,
            description="SVAMP Dataset",
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
                "body": datasets.Value("string"),
                "ques": datasets.Value("string"),
                "numbers": datasets.Value("string"),
                "equation": datasets.Value("string"),
                "group_nums": datasets.features.Sequence(datasets.Value("string")),
                "type": datasets.Value("string"),
                "variation type": datasets.Value("string"),
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
        ]

    def _generate_examples(self, filepath: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, ex in enumerate(reader):
                # Missing key in train set
                if "type" not in ex:
                    ex["Type"] = None
                    ex["Variation Type"] = None
                # Replace numbers into the question
                numbers = ex["Numbers"].split()
                for num_idx, num in enumerate(numbers):
                    ex["Question"] = ex["Question"].replace(f"number{num_idx}", num)
                # Capitalize the first letter of the question
                ex["Question"] = ex["Question"].replace(" .", ".").replace(" ?", "?")
                sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                ex["Question"] = " ".join([sent.capitalize() for sent in sent_tokenizer.tokenize(ex["Question"])])
                yield idx, {
                    "id": str(idx),
                    "question": ex["Question"],
                    "answer": ex["Answer"],
                    "body": ex["Body"],
                    "ques": ex["Ques"],
                    "numbers": ex["Numbers"],
                    "equation": ex["Equation"],
                    "group_nums": list(ex["group_nums"]),
                    "type": ex["Type"],
                    "variation type": ex["Variation Type"],
                }
