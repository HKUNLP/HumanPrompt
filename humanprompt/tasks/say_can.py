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
"""Do As I Can, Not As I Say: Grounding Language in Robotic Affordances"""
# mypy: ignore-errors

import datasets

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{saycan2022arxiv,
    title={Do As I Can and Not As I Say: Grounding Language in Robotic Affordances},
    author={Michael Ahn and Anthony Brohan and Noah Brown and Yevgen Chebotar and Omar Cortes and Byron David and Chelsea Finn and Chuyuan Fu and Keerthana Gopalakrishnan and Karol Hausman and Alex Herzog and Daniel Ho and Jasmine Hsu and Julian Ibarz and Brian Ichter and Alex Irpan and Eric Jang and Rosario Jauregui Ruano and Kyle Jeffrey and Sally Jesmonth and Nikhil Joshi and Ryan Julian and Dmitry Kalashnikov and Yuheng Kuang and Kuang-Huei Lee and Sergey Levine and Yao Lu and Linda Luu and Carolina Parada and Peter Pastor and Jornell Quiambao and Kanishka Rao and Jarek Rettinghouse and Diego Reyes and Pierre Sermanet and Nicolas Sievers and Clayton Tan and Alexander Toshev and Vincent Vanhoucke and Fei Xia and Ted Xiao and Peng Xu and Sichun Xu and Mengyuan Yan and Andy Zeng},
    booktitle={arXiv preprint arXiv:2204.01691},
    year={2022}
}
"""

_DESCRIPTION = """\
Large language models can encode a wealth of semantic knowledge about the world. Such knowledge could in principle be extremely useful to robots aiming to act upon high-level, temporally extended instructions expressed in natural language. However, a significant weakness of language models is that they lack contextual grounding, which makes it difficult to leverage them for decision making within a given real-world context. For example, asking a language model to describe how to clean a spill might result in a reasonable narrative, but it may not be applicable to a particular agent, such as a robot, that needs to perform this task in a particular environment. We propose to provide this grounding by means of pretrained behaviors, which are used to condition the model to propose natural language actions that are both feasible and contextually appropriate. The robot can act as the language model’s “hands and eyes,” while the language model supplies high-level semantic knowledge about the task. We show how low-level tasks can be combined with large language models so that the language model provides high-level knowledge about the procedures for performing complex and temporally extended instructions, while value functions associated with these tasks provide the grounding necessary to connect this knowledge to a particular physical environment. We evaluate our method on a number of real-world robotic tasks, where we show that this approach is capable of completing long-horizon, abstract, natural language instructions on a mobile manipulator.
"""

_HOMEPAGE = "https://say-can.github.io/"

_LICENSE = "Apache-2.0"

_URL = "https://raw.githubusercontent.com/say-can/say-can.github.io/main/data/saycan_plan_v0_l.tsv"


class SayCan(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="saycan",
            version=VERSION,
            description="Large language models can encode a wealth of semantic knowledge about the world. Such knowledge could in principle be extremely useful to robots aiming to act upon high-level, temporally extended instructions expressed in natural language. However, a significant weakness of language models is that they lack contextual grounding, which makes it difficult to leverage them for decision making within a given real-world context. For example, asking a language model to describe how to clean a spill might result in a reasonable narrative, but it may not be applicable to a particular agent, such as a robot, that needs to perform this task in a particular environment. We propose to provide this grounding by means of pretrained behaviors, which are used to condition the model to propose natural language actions that are both feasible and contextually appropriate. The robot can act as the language model’s “hands and eyes,” while the language model supplies high-level semantic knowledge about the task. We show how low-level tasks can be combined with large language models so that the language model provides high-level knowledge about the procedures for performing complex and temporally extended instructions, while value functions associated with these tasks provide the grounding necessary to connect this knowledge to a particular physical environment. We evaluate our method on a number of real-world robotic tasks, where we show that this approach is capable of completing long-horizon, abstract, natural language instructions on a mobile manipulator.",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

    def _info(self):
        features = datasets.Features(
            {
                "input": datasets.Value("string"),
                "output": datasets.Sequence(datasets.Value("string")),
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

    def _split_generators(self, dl_manager):
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_filepath": downloaded_filepath},
            ),
        ]

    def _generate_examples(self, data_filepath):
        """This function returns the examples in the raw (text) form."""
        # read in the tsv data
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                # skip the header
                if id_ == 0:
                    continue
                row = row.strip().split("\t")
                input, output = row
                yield id_, {"input": input, "output": output.strip(".").split(", ")}
