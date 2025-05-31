import re
import time
from typing import Tuple, List
import os
from typing import Tuple

from autoplc_scl.tools import PromptResultUtil
from autoplc_scl.tools import APIDataLoader
from autoplc_scl.agents.clients import ClientManager, OpenAIClient
import logging

logger = logging.getLogger("autoplc_scl")


class Modeler():
    """
    The Modeler class is used to generate algorithm process descriptions. It generates an algorithmic process that guides subsequent code generation by running modeling tasks.

    Attributes:
        base_logs_folder (str): The path used to store the base log folder.

    Methods:
        run_modeling_task(task: dict, retrieved_examples: List[dict], related_algorithm: List[dict]) -> Tuple[str, int, int]:
            Run the modeling task to generate an algorithm process description. This method uses task information, retrieved examples, and related algorithms to generate an algorithm flow description.
    """

    base_logs_folder = None

    @classmethod
    def run_modeling_task(cls, 
            task: dict,
            retrieved_examples: List[dict],
            related_algorithm: List[dict],
            openai_client : OpenAIClient,
            load_few_shots: bool = True,
            ) -> str:
        """
        Run the modeling task to generate an algorithm process description.

        parameter:
        -task: A dictionary containing task information.
        -retrieved_examples: The retrieved list of examples, each example is a dictionary.
        -related_algorithm: A list of related algorithms, each algorithm is a dictionary.
        -openai_client : LLM client
        -load_few_shots: Whether to load the few-shot example.
        
        return:
        -algorithm_for_this_task: The generated algorithm flow description string.
        """

        logger.info("Start executing planner")
        # Record the start time, used to calculate the execution time
        start_time = time.time()
        logger.info("\n------------------------------------------------\n")

        # Construct a message list, including system prompts and user requirements
        messages = [ 
            # {"role": "system", "content": state_machine_prompt}, 
            # {"role": "system", "content": system_prompt_state_machine}
            {"role": "system", "content": state_machine_prompt_en}
            # {"role": "user", "content": str(task)}
        ]

        if load_few_shots:
            fewshots = cls.load_plan_gen_shots(retrieved_examples, related_algorithm)
            messages.extend(fewshots)

        messages.append({"role": "user", "content": plan_shots_prompt_en.format(task_requirements = str(task)) })
        # print(messages)

        # Calling model generation planning
        response = openai_client.call(
            messages,
            task_name=task["name"],
            role_name="planner"
        )

        # Print the end-of-print generation plan message and execution time
        # print("\n------------------------------------------------\n")
        # print("The planner has been executed")

        algo = cls.extract_content(response)

        # Returns the planning content and the number of token usage
        logger.info(f"End - generate plan - Execution Time: {(time.time() - start_time):.6f} seconds")

        return algo

    @classmethod
    def extract_content(cls,response) -> str:
        """
        This function is used to extract the generated content from the response.

        parameter:
        -response: The response object of the AI ​​model.

        return:
        -algo: Extracted algorithm flow description string.
        """
        if hasattr(response, 'choices'):
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                algo = choice.message.content
            else:
                algo = choice['message']['content']
        else:
            algo = response.content[0].text

        return algo

    @classmethod
    def load_plan_gen_shots(cls, examples: List[dict], algorithms: List[str]):
        """
        Load few-shot for plan
        """
        formatted = []
        shot_prompt = plan_shots_prompt_en
        for example, algo in zip(examples, algorithms):
            icl_input = shot_prompt.format(
                task_requirements=str(example['json_desc'])
                # algorithm_process = "",
                # algorithm_process=f"<! -- A plan that you can refer to when coding -->\n<Plan>\n{algo}\n</Plan>\n"
            )
            icl_output = PromptResultUtil.remove_braces(algo)
            formatted.append({"role": "user", "content": icl_input})
            formatted.append({"role": "assistant", "content": icl_output})
        # print("---------------------------------------------------------------------------------------------------------------------")
        # print(formatted)
        return formatted



plan_shots_prompt_en = """Here is the input, structured in XML format:
<! -- SCL programming task requirements to be completed -->
<Task_Requirements>
{task_requirements}
</Task_Requirements>
"""

state_machine_prompt_en = """
You are a PLC engineer.

Please first determine whether the given requirement is a **process control task** or a **data processing task**.

For **process control tasks**:
    1. Analyze the possible states.
    2. Identify the state transition events.
    3. Describe the algorithmic workflow. Do not output pseudocode or any form of code!

For **data processing tasks**:
    1. Describe the algorithmic workflow. Do not output pseudocode or any form of code!

Notes:
- Maintain a professional and rigorous tone, in line with Siemens S7-1200/1500 PLC standards.
- Be cautious with initialization steps to avoid unintentional deletion of important data.
- If no transition event occurs, the current state's actions should be maintained.
- If the requirement explicitly calls for exception handling, include it; otherwise, it is not necessary.
""".strip()

