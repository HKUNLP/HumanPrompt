# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors, The Google AI Language Team Authors and the current dataset script contributor.
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
"""The WikiTableQuestions dataset is for the task of question answering on semi-structured HTML tables"""
# mypy: ignore-errors
import json
import os
from typing import Iterator, List, Tuple

import datasets


def _load_table(table_path: str, page_title_path: str = None) -> dict:
    """
    attention: the table_path must be the .tsv path.
    Load the WikiTableQuestion from csv file. Result in a dict format like:
    {"header": [header1, header2,...], "rows": [[row11, row12, ...], [row21,...]... [...rownm]]}
    """

    def __load_table(table_path: str) -> dict:
        """
        attention: the table_path must be the .tsv path.
        Load the WikiTableQuestion from csv file. Result in a dict format like:
        {"header": [header1, header2,...], "rows": [[row11, row12, ...], [row21,...]... [...rownm]]}
        """

        def __extract_content(_line: str) -> List:
            _vals = [
                _.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")
            ]
            return _vals

        with open(table_path, "r") as f:
            lines = f.readlines()

            rows = []
            for i, line in enumerate(lines):
                line = line.strip("\n")
                if i == 0:
                    header = line.split("\t")
                else:
                    rows.append(__extract_content(line))

        table_item = {"header": header, "rows": rows}

        # Defense assertion
        for i in range(len(rows) - 1):
            if not len(rows[i]) == len(rows[i - 1]):
                raise ValueError("some rows have diff cols.")

        return table_item

    table_item = __load_table(table_path)

    # Load page title
    if not page_title_path:
        page_title_path = table_path.replace("csv", "page").replace(".tsv", ".json")
    with open(page_title_path, "r") as f:
        page_title = json.load(f)["title"]
    table_item["page_title"] = page_title

    return table_item


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{pasupat-liang-2015-compositional,
    title = "Compositional Semantic Parsing on Semi-Structured Tables",
    author = "Pasupat, Panupong  and
      Liang, Percy",
    booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = jul,
    year = "2015",
    address = "Beijing, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P15-1142",
    doi = "10.3115/v1/P15-1142",
    pages = "1470--1480",
}
"""

_DESCRIPTION = """\
Two important aspects of semantic parsing for question answering are the breadth of the knowledge source and the depth of
logical compositionality. While existing work trades off one aspect for another, this paper simultaneously makes progress
on both fronts through a new task: answering complex questions on semi-structured tables using question-answer pairs as
supervision. The central challenge arises from two compounding factors: the broader domain results in an open-ended set
of relations, and the deeper compositionality results in a combinatorial explosion in the space of logical forms. We
propose a logical-form driven parsing algorithm guided by strong typing constraints and show that it obtains significant
 improvements over natural baselines. For evaluation, we created a new dataset of 22,033 complex questions on Wikipedia
  tables, which is made publicly available.
"""

_HOMEPAGE = "https://ppasupat.github.io/WikiTableQuestions/"

_LICENSE = "CC-BY-SA-4.0 License"

_URL = "https://github.com/ppasupat/WikiTableQuestions/archive/refs/heads/master.zip"


class WikiTableQuestion(datasets.GeneratorBasedBuilder):
    """The WikiTableQuestions dataset"""

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "table_id": datasets.Value("string"),
                    "table": {
                        "page_title": datasets.Value("string"),
                        "header": datasets.features.Sequence(datasets.Value("string")),
                        "rows": datasets.features.Sequence(
                            datasets.features.Sequence(datasets.Value("string"))
                        ),
                    },
                    "answer_text": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = os.path.join(
            dl_manager.download_and_extract(_URL), "WikiTableQuestions-master"
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/random-split-1-train.tsv"),
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data/random-split-1-dev.tsv"),
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data/pristine-unseen-tables.tsv"
                    ),
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(
        self, filepath: str, data_dir: str
    ) -> Iterator[Tuple[int, dict]]:
        """Yields examples."""
        # data_id, question, table_id, gold_result_str
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # skip the header
                if idx == 0:
                    continue
                data_id, question, table_id, gold_result_str = line.strip("\n").split(
                    "\t"
                )
                gold_result = gold_result_str.split("|")
                yield idx, {
                    "id": data_id,
                    "question": question,
                    "table_id": table_id,
                    "table": _load_table(
                        os.path.join(data_dir, table_id.replace(".csv", ".tsv"))
                    ),
                    # convert the .csv postfix to .tsv, for easier read-in
                    "answer_text": gold_result,
                }
