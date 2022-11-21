import sqlite3
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import sqlparse

from .transform_base import Transform


class DBTransform(Transform):
    @staticmethod
    def transform(
        x: Union[str, Dict], y: Union[str, Dict] = None, **kwargs: Any
    ) -> str:
        """
        Transform the db into a prompt.

        Args:
            x: input, contains the key of 'db'
            y: not required
            **kwargs:

        Returns: the prompt of database.

        """
        assert isinstance(x, Dict)

        db_prompt = build_db_prompt(
            x["db"],
            kwargs["prompt_style"]
            if "prompt_style" in kwargs
            else "create_table_select_3",
        )
        return db_prompt


def convert_to_df(table: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
    """
    Convert table to pandas DataFrame.

    Args:
        table: a pandas DataFrame or a dict

    Returns: a pandas DataFrame

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
    table: Union[pd.DataFrame, Dict], table_name: str = "table"
) -> str:
    """
    Return the CREATE TABLE clause as prompt.

    An example output is:

    CREATE TABLE 2007_new_orleans_saints_season(
    row_id int,
    week int,
    date text,
    opponent text,
    time text,
    game site text,
    tv text,
    result/score text,
    record text)

    Args:
        table: the table to be converted to a prompt
        table_name: the name of the table

    Returns: the prompt(create part)

    """
    df = convert_to_df(table)

    string = "CREATE TABLE {}(\n".format(table_name)
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
    table_name: str,
    prompt_style: str,
    n_rows: int = None,
    seperator: str = "tab",
) -> str:
    """
    Return the first n_rows of rows from table contents as prompt.

    An example output is:

    /*
    3 example rows from table 2007_new_orleans_saints_season:
    SELECT * FROM w LIMIT 3;
    row_id	week	date	opponent	time	game site	tv	result/score	record
    0	1	2007-9-6	indianapolis colts	t20:30 edt	away	nbc	loss	0–1
    1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	home	fox	win	1-1
    2	3	2007-9-24	tennessee titans	t20:30 edt	away	espn	loss	1-2
    */

    Args:
        table: the table to be converted to a prompt
        table_name: the name of the table
        prompt_style: the style of the prompt
        n_rows: the number of rows to be included in the prompt
        seperator: the seperator between columns

    Returns: the prompt(select and content part)

    """

    df = convert_to_df(table)
    select_head = ""

    if prompt_style == "create_table_select_full_table":
        # Select the full table
        n_rows = len(df)
        select_head += f"/*\nAll rows from table {table_name}:\nSELECT * FROM w;\n"
    elif prompt_style in ["create_table_select_3", "create_table_select_n"]:
        # Select the first 3/n rows
        if prompt_style == "create_table_select_3":
            n_rows = 3
        else:
            assert (
                n_rows
            ), 'When using prompt style "create_table_select_n", parameter n_rows must be specified.'
        select_head += f"/*\n{n_rows} example rows from table {table_name}:\nSELECT * FROM w LIMIT {n_rows};\n"
    elif prompt_style == "no_table":
        # No table input
        n_rows = 0
    else:
        raise ValueError("prompt_style not supported")

    content = ""

    if seperator == "tab":
        for column_id, header in enumerate(df.columns):
            content += str(header)
            if column_id != len(df.columns) - 1:
                content += "\t"
        content += "\n"

        for row_id, row in df.iloc[:n_rows].iterrows():
            for column_id, header in enumerate(df.columns):
                content += str(row[header])
                if column_id != len(df.columns) - 1:
                    content += "\t"
            content += "\n"
        content += "*/\n"
    elif seperator == "spaces":
        # Then we can just use the default pandas to_string() method
        content += df[:n_rows].to_string(index=False)
    else:
        raise ValueError(f"seperator {seperator} not supported")

    return select_head + content


def build_table_prompt(
    table: Union[pd.DataFrame, Dict],
    table_name: str = "table",
    prompt_style: str = "create_table_select_3",
) -> str:
    """
    Build a prompt for a table. Table can be a pandas dataframe or a dict,
    when it is a dict, it should be formatted as {"header": list, "rows": list_of_list},
    follow the same practice of TaPEx(https://arxiv.org/abs/2107.07653) and UnifiedSKG(https://arxiv.org/abs/2201.05966).

    An example output is:

    CREATE TABLE 2007_new_orleans_saints_season(
    row_id int,
    week int,
    date text,
    opponent text,
    time text,
    game site text,
    tv text,
    result/score text,
    record text)
    /*
    3 example rows from table 2007_new_orleans_saints_season:
    SELECT * FROM w LIMIT 3;
    row_id	week	date	opponent	time	game site	tv	result/score	record
    0	1	2007-9-6	indianapolis colts	t20:30 edt	away	nbc	loss	0–1
    1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	home	fox	win	1-1
    2	3	2007-9-24	tennessee titans	t20:30 edt	away	espn	loss	1-2
    */

    Args:
        table: the table to be converted to a prompt
        table_name: the name of the table
        prompt_style: the style of the prompt

    Returns:

    """
    table_prompt = ""

    table_prompt += build_db_create_table_prompt_part(table, table_name)
    table_prompt += build_db_select_x_prompt_part(table, table_name, prompt_style)

    return table_prompt


def build_db_prompt(
    db: Union[List[Tuple[str, Union[pd.DataFrame, Dict]]], str],
    prompt_style: str = "create_table_select_3",
) -> str:
    """
    Build a prompt for a database. Database can be a list of tuple (table_name, table) or a path to a sqlite file.
    The db will provide more information of primary key, foreign key, and unique key, while the list of tuple will not(in this implementation).

    Args:
        db: the database to be converted to a prompt, can be a list of tuple (table_name, table) or a path to a sqlite file
        prompt_style: the style of the prompt

    An example output is:
    CREATE TABLE Highschooler(
    ID int primary key,
    name text,
    grade int)
    /*
    3 example rows:
    SELECT * FROM Highschooler LIMIT 3;
    ID name grade
    1510 Jordan 9
    1689 Gabriel 9
    1381 Tiffany 9
    */
    CREATE TABLE Friend(
    student_id int,
    friend_id int,
    primary key (student_id,friend_id),
    foreign key(student_id) references Highschooler(ID),
    foreign key (friend_id) references Highschooler(ID)
    )
    /*
    3 example rows:
    SELECT * FROM Friend LIMIT 3;
    student_id friend_id
    1510 1381
    1510 1689
    1689 1709
    */
    CREATE TABLE Likes(
    student_id int,
    liked_id int,
    primary key (student_id, liked_id),
    foreign key (liked_id) references Highschooler(ID),
    foreign key (student_id) references Highschooler(ID)
    )
    /*
    3 example rows:
    SELECT * FROM Likes LIMIT 3;
    student_id liked_id
    1689 1709
    1709 1689
    1782 1709
    */

    Returns: the prompt

    """

    def _get_db_schemas(db_path: str) -> Dict[str, str]:
        schemas = {}
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                cursor.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
                        table[0]
                    )
                )
                schemas[table[0]] = cursor.fetchone()[0]
        return schemas

    def _get_db_content(db_path: str) -> Dict[str, pd.DataFrame]:
        db_path = f"file:{db_path}?mode=ro"
        results = {}
        with sqlite3.connect(db_path, uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                cursor.execute("PRAGMA table_info({})".format(table[0]))
                results[table[0]] = pd.read_sql_query(f"SELECT * FROM {table[0]}", conn)
        return results

    if isinstance(db, str):
        # We suppose at this time, db is the sqlite file path
        schemas = _get_db_schemas(db)
        contents = _get_db_content(db)
        table_prompt = ""

        for table_name in schemas.keys():
            # build_db_create_table_prompt_part
            table_prompt += sqlparse.format(schemas[table_name], reindent_aligned=False)
            table_prompt += "\n"
            # build_db_select_x_prompt_part
            table_prompt += build_db_select_x_prompt_part(
                contents[table_name], table_name, prompt_style
            )
    else:
        table_prompt = ""
        for table_name, table in db:
            table_prompt += build_table_prompt(table, table_name, prompt_style)
            table_prompt += "\n"

    return table_prompt
