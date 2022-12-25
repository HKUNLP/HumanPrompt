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
"""MultiArith Dataset"""
# mypy: ignore-errors
from typing import Any, Dict, Iterator, List, Tuple
import json

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{roy-roth-2015-solving,
    title = "Solving General Arithmetic Word Problems",
    author = "Roy, Subhro  and
      Roth, Dan",
    booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D15-1202",
    doi = "10.18653/v1/D15-1202",
    pages = "1743--1752",
}
"""
_DESCRIPTION = """
MultiArith dataset in MAWPS.
"""
_HOMEPAGE = ""
_LICENSE = "MIT License"
_URL = (
    "https://raw.githubusercontent.com/kojima-takeshi188/zero_shot_cot/main/dataset/MultiArith/"
)
_DEV_FILE = "MultiArith.json"
_URLS = {
    "dev": f"{_URL}{_DEV_FILE}",
}


class SVAMP(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="multi_arith",
            version=VERSION,
            description="MultiArith Dataset",
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
                "sQuestion": datasets.Value("string"),
                "lSolutions": datasets.Value("string"),
                "lEquations": datasets.features.Sequence(datasets.Value("string")),
                "lAlignments": datasets.features.Sequence(datasets.Value("string")),
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
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
        ]

    def _generate_examples(self, filepath: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for idx, ex in enumerate(data):
                    yield idx, {
                        "id": str(ex["iIndex"]),
                        "sQuestion": ex["sQuestion"],
                        "lSolutions": "|".join([str(s) for s in ex["lSolutions"]]),
                        "lEquations": ex["lEquations"],
                        "lAlignments": ex["lAlignments"],
                    }
