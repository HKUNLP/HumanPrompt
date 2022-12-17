import re
from typing import List, Tuple, Union

import sqlparse


class TreeNode(object):
    def __init__(self, name: str = None, father: object = None, line: int = -1) -> None:
        super(TreeNode, self).__init__()
        if father is not None and not isinstance(father, TreeNode):
            raise TypeError("father should be TreeNode")
        self.name: str = name
        self.rename: str = name
        self.line: int = line
        self.father: TreeNode = father
        self.children: List = []

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.rename == other.rename

    def __hash__(self) -> int:
        return hash(self.rename)

    def set_name(self, name: str) -> None:
        self.name = name
        self.rename = name

    def add_child(self, child: object) -> None:
        if not isinstance(child, TreeNode):
            raise TypeError("child should be TreeNode")
        self.children.append(child)
        child.father = self

    def rename_father(self, var_name: str) -> None:
        self.father.rename = self.father.rename.replace(
            self.name, "{}".format(var_name)
        )


def parse_program(binder_program: str, symbol: str = "QA") -> TreeNode:
    """
    Parse 'symbol'() into a tree for execution guiding.
    :param binder_program:
    :param symbol:
    :return:
    """

    stack: List = []  # Saving the state of the char.
    expression_stack: List = []  # Saving the state of the expression.
    current_tree_node = TreeNode(name=binder_program)
    line_idx = 0

    for idx in range(len(binder_program)):
        if binder_program[idx] == "\n":
            line_idx += 1
        if binder_program[idx] == "(":
            stack.append(idx)
            if (
                idx > 1
                and binder_program[idx - len(symbol) : idx + 1] == "{}(".format(symbol)
                and idx - len(symbol) != 0
            ):
                tree_node = TreeNode(line=line_idx)
                current_tree_node.add_child(tree_node)
                expression_stack.append(current_tree_node)
                current_tree_node = tree_node
        elif binder_program[idx] == ")":
            left_clause_idx = stack.pop()
            if (
                idx > 1
                and binder_program[left_clause_idx - len(symbol) : left_clause_idx + 1]
                == "{}(".format(symbol)
                and left_clause_idx - len(symbol) != 0
            ):
                nsql_span = binder_program[left_clause_idx - len(symbol) : idx + 1]
                current_tree_node.set_name(nsql_span)
                current_tree_node = expression_stack.pop()

    return current_tree_node


def get_steps(tree_node: TreeNode, steps: List) -> None:
    """
    Pred-Order Traversal
    :param tree_node: the root node of the tree
    :param steps: the inputted list to save the steps
    :return: None, the steps is saved in the steps variable inputted.
    """
    for child in tree_node.children:
        get_steps(child, steps)
    steps.append(tree_node)


def parse_question_paras(binder_part: str, symbol: str = "QA") -> Tuple:
    """
    Parse the binder_part of a binder_program into the question and a list of parameters.
    :param binder_part:
    :param symbol:
    :return:
    """
    # We assume there's no nested 'symbol' inside when running this func
    binder_part = binder_part.strip(" ;")
    assert (
        binder_part[: len(symbol) + 1] == "{}(".format(symbol)
        and binder_part[-1] == ")"
    ), "must start with {}( symbol and end with )".format(symbol)
    assert (
        symbol not in binder_part[len(symbol) : -1]
    ), "must have no nested {} inside".format(symbol)

    # Get question and the left part(paras_raw_str)
    all_quote_idx = [i.start() for i in re.finditer('"', binder_part)]
    question = binder_part[all_quote_idx[0] + 1 : all_quote_idx[1]]
    paras_raw_str = binder_part[all_quote_idx[1] + 1 : -1].strip(" ;")

    # Split Parameters(SQL/column/value) from all parameters.
    paras = [_para.strip(" ;") for _para in sqlparse.split(paras_raw_str)]
    return question, paras


def remove_duplicate(original_list: List) -> List:
    """
    Remove the duplicate elements in the list.
    :param original_list: the original list
    :return:
    """
    no_duplicate_list = []
    for i in original_list:
        if i not in no_duplicate_list:
            no_duplicate_list.append(i)
    return no_duplicate_list


def get_variable_and_type(line: str, binder_part: str) -> Union[str, None]:
    assert binder_part in line, "The step name {} is not in the line {}".format(
        binder_part, line
    )
    for _idx in range(len(line)):
        if line[_idx : _idx + len(binder_part)] == binder_part:
            previous_part = line[0:_idx]
            return previous_part
    return None
