from typing import Dict, Union

import pandas as pd


def convert_to_df(table: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
    """
    Convert table to pandas DataFrame.
    """
    # if table is a dict, convert it to a dataframe
    if isinstance(table, dict):
        df = pd.DataFrame(data=table["rows"], columns=table["header"])
    elif isinstance(table, pd.DataFrame):
        df = table
    else:
        raise TypeError("table should be a dict or a pandas dataframe")

    return df


def build_db_create_table_prompt_part(
    table: Union[pd.DataFrame, Dict], title: str = ""
) -> str:
    """
    Return the CREATE TABLE clause as prompt.
    """
    df = convert_to_df(table)

    string = "CREATE TABLE {}(\n".format(title)
    for header in df.columns:
        column_type = "text"
        try:
            if df[header].dtype == "int64":
                column_type = "int"
            elif df[header].dtype == "float64":
                column_type = "real"
            elif df[header].dtype == "datetime64":
                column_type = "datetime"
        except AttributeError as e:
            raise Exception(e)

        string += "\t{} {},\n".format(header, column_type)
    string = string.rstrip(",\n") + ")\n"
    return string


def build_db_select_x_prompt_part(
    table: Union[pd.DataFrame, Dict],
    prompt_style: str,
) -> str:
    """
    Return the first X rows table contents as prompt.
    """

    df = convert_to_df(table)
    select_head = ""

    if prompt_style == "create_table_select_full_table":
        num_rows = len(df)
        select_head += "/*\nAll rows of the table:\nSELECT * FROM w;\n"
    elif prompt_style == "create_table_select_3":
        num_rows = 3
        select_head += "/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n".format(
            num_rows, num_rows
        )
    elif prompt_style == "no_table":
        # No table input, to test Codex QA with only internal knowledge
        num_rows = 0
    else:
        raise ValueError("prompt_style not supported")

    content = ""
    for column_id, header in enumerate(df.columns):
        content += str(header)
        if column_id != len(df.columns) - 1:
            content += "\t"
    content += "\n"

    for row_id, row in df.iloc[:num_rows].iterrows():
        for column_id, header in enumerate(df.columns):
            content += str(row[header])
            if column_id != len(df.columns) - 1:
                content += "\t"
        content += "\n"
    content += "*/\n"

    return select_head + content


def build_db_prompt(
    table: Union[pd.DataFrame, Dict],
    title: str = "",
    prompt_style: str = "create_table_select_3",
) -> str:
    table_prompt = ""

    table_prompt += build_db_create_table_prompt_part(table, title)
    table_prompt += build_db_select_x_prompt_part(table, prompt_style)

    return table_prompt
