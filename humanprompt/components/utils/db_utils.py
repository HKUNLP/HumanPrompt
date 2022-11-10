import pandas as pd
from typing import Any, Dict, Union


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


def build_db_create_table_prompt_part(table: Union[pd.DataFrame, Dict], title: str = "") -> str:
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


def build_db_select_x_prompt_part(table: Union[pd.DataFrame, Dict], prompt_style: str) -> str:
    """
    Return the first X rows table contents as prompt.
    """

    df = convert_to_df(table)

    if prompt_style == "create_table_select_full_table":
        num_rows = len(df)
        string = "/*\nAll rows of the table:\nSELECT * FROM w;\n"
    elif prompt_style == "create_table_select_3":
        num_rows = 3
        string = "/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n".format(
            num_rows, num_rows
        )
    else:
        raise ValueError("prompt_style not supported")

    for column_id, header in enumerate(df.columns):
        string += str(header)
        if column_id != len(df.columns) - 1:
            string += "\t"
    string += "\n"

    for row_id, row in df.iloc[:num_rows].iterrows():
        for column_id, header in enumerate(df.columns):
            string += str(row[header])
            if column_id != len(df.columns) - 1:
                string += "\t"
        string += "\n"
    string += "*/\n"

    return string


def build_db_prompt(table: Union[pd.DataFrame, Dict], title: str = "",
                    prompt_style: str = "create_table_select_3") -> str:
    table_prompt = ""

    table_prompt += build_db_create_table_prompt_part(table, title)
    table_prompt += build_db_select_x_prompt_part(table, prompt_style)

    return table_prompt
