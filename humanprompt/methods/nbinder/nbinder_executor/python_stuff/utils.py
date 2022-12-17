import keyword
from typing import Sequence


def rename_variable_if_keyword(
    variable: str, keyword_list: Sequence[str] = keyword.kwlist
) -> str:
    """
    Rename the variable if it contains keyword.
    :param variable:
    :param keyword_list:
    :return:
    """
    for _keyword in keyword_list:
        if _keyword == variable:
            variable = variable.replace(_keyword, f"{_keyword}_")
    return variable
