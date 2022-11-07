import os

import pytest


@pytest.fixture
def openai_api_key():
    assert os.environ.get("OPENAI_API_KEY")
