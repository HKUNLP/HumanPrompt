import os
import pickle
import uuid
from copy import deepcopy
from subprocess import PIPE, Popen
from typing import Dict, List, Tuple

from .neural_module import NeuralModule
from .parser import (
    TreeNode,
    get_steps,
    get_variable_and_type,
    parse_program,
    parse_question_paras,
    remove_duplicate,
)
from .python_stuff.utils import rename_variable_if_keyword


class Executor(object):
    def __init__(self, **kwargs: Dict) -> None:
        self.tmp_var_idx = 0  # The index of the tmp variable.
        self.neural_model = NeuralModule(
            **kwargs
        )  # The neural model to execute the neural part.

    def execute(
        self, binder_program: str, context: Dict, verbose: bool = True
    ) -> Tuple[bool, str, str, list]:
        """
        Execute the binder program.
        Args:
            binder_program: The binder program to execute.
            context: context variables.
            verbose: Whether to print the execution process.

        Returns: The result of the binder program executed on context.
        todo: extend to more programming languages

        """
        steps: List[TreeNode] = []
        all_context = deepcopy(context)
        root_node = parse_program(
            binder_program
        )  # Parse execution tree from binder_program.
        get_steps(root_node, steps)  # Flatten the execution tree and get the steps.
        steps = remove_duplicate(steps)  # Remove the duplicate steps.
        paras_used: List[str] = []
        sub_qas = []  # For log and train the neural module.

        for step in steps[:-1]:
            line: str = binder_program.split("\n")[step.line]
            variable_and_type = get_variable_and_type(
                line, step.name
            )  # if QA is nested, it will not work
            if len(variable_and_type.split(":")) == 2:
                # if there is no nested QA, like "age: int = QA("What is the age of the person?", Jack)"
                _variable, _type = (
                    variable_and_type.split(":")[0].strip(),
                    variable_and_type.split(":")[1].strip(),
                )
                _variable = rename_variable_if_keyword(_variable)
                variable_and_type = _variable + " : " + _type
            else:
                # if there is nested QA, like:
                # "wife_age: int = QA("What is the age of the person?", QA("What is the wife of the person?", Jack))"
                _type = "Any"
                _variable = "tmp_var_{}".format(self.tmp_var_idx)
                self.tmp_var_idx += 1
                variable_and_type = _variable + " : " + _type

            # In case the variable name contains keyword.
            neural_part = step.rename
            question, paras = parse_question_paras(neural_part)
            x = {
                "question": question,
                "variable_and_type": variable_and_type,
                **{para: all_context[para] for para in paras},
            }
            neural_part_result = self.neural_model.run(x)
            all_context[_variable] = neural_part_result  # add the result to the context
            assert isinstance(neural_part_result, str), "neural_part_result must be str"

            sub_qas.append(
                {
                    "question": question,
                    "variable_and_type": variable_and_type,
                    "neural_part_result": neural_part_result,
                }
            )

            # todo: we can change this part into a smaller model.
            step.rename_father(neural_part_result)

            if verbose:
                print("Step: {} ".format(step.name))
                print("Context: {}".format({para: all_context[para] for para in paras}))
                print("Result: {}".format(neural_part_result))

        # Execute the last step. Pure symbolic code.
        def _insert_context_variables(_program: str, context_variables: Dict) -> str:
            _tmp_root_path = "tmp"
            os.makedirs(_tmp_root_path, exist_ok=True)
            for variable_name, variable_value in context_variables.items():
                stored_path = os.path.join(
                    _tmp_root_path, "{}_{}.pkl".format(variable_name, uuid.uuid4())
                )
                pickle.dump(variable_value, open(stored_path, "wb"))
                _program = (
                    'obj = open("{}", "rb")\n{} = pickle.load(obj)\n'.format(
                        stored_path, variable_name
                    )
                    + _program
                )
            return _program

        # Add context variables and dependencies to the program.
        program = _insert_context_variables(
            steps[-1].rename, {para: all_context[para] for para in set(paras_used)}
        )

        tmp_root_path = "tmp_python"
        os.makedirs(tmp_root_path, exist_ok=True)
        python_file_path = "{}.py".format(format(uuid.uuid4()))
        python_path = os.path.join(tmp_root_path, python_file_path)
        with open(python_path, "w") as f:
            f.write(program)

        # Auto import packages used and delete packages not used.
        p_import = Popen(["autoimport", python_path], stdout=PIPE, stderr=PIPE)
        p_import.communicate()

        # Clean the code by autoflake.
        p_clean = Popen(
            [
                "autoflake",
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                python_path,
            ],
            stdout=PIPE,
            stderr=PIPE,
        )
        p_clean.communicate()
        pure_program = open(python_path).read()
        if verbose:
            print("Clean the code by autoflake.")
            print("After clean:")
            print(pure_program)

        # Execute the code.
        p = Popen("python " + python_path, shell=True, stdout=PIPE, stderr=PIPE)
        try:
            stdout, stderr = p.communicate(timeout=10)
        except Exception:
            # The process took longer than 10 seconds, so kill it
            p.kill()
            stdout, stderr = "".encode("utf-8"), "Timeout".encode("utf-8")

        if stderr:  # Error in execution so that we didn't get result.
            return False, stderr.decode("utf-8"), pure_program, sub_qas

        return True, stdout.decode("utf-8"), pure_program, sub_qas  # Read the result
