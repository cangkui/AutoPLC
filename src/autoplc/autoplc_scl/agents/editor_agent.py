from typing import Tuple, List
from autoplc_scl.tools.prompt_res_util import PromptResultUtil
from autoplc_scl.tools import APIDataLoader
from autoplc_scl.agents.clients import ClientManager, OpenAIClient
import logging
logger = logging.getLogger("autoplc_scl")

class LogicComposer():
    """
    The LogicComposer class is used to generate SCL code. It provides the ability to extract code from the response and generate code by running the editor agent.

    Attributes:
        base_logs_folder (str): The path used to store the underlying log folder.

    Methods:
        extract_code_from_response(text: str) -> str:
            Extract SCL code from the given text response.

        run_gen_scl(task: dict, retrieved_examples: List[dict], related_algorithm: List[dict], algorithm_for_this_task: str, model: str) -> str:
            Run the editor agent to generate SCL code. The method generates code using task information, retrieved examples, related algorithms, and algorithm descriptions for specific tasks.
    """

    # The path used to store the underlying log folder
    base_logs_folder: str = None

    @classmethod
    def extract_code_from_response(cls, text: str) -> str:
        """
        Extract code from response
        ```scl
            <Code>
        ```
        """
        import re
        matches = re.findall(r'```scl\n(.*?)\n```', text, re.DOTALL)
        if matches:
            # Splice all ST blocks, separate them with line breaks in the middle
            return "\n\n".join(m.strip() for m in matches)
        else:
            return text.strip()  # Fallback: Return to the original text directly
    
    @classmethod
    def run_gen_scl(cls,
            task: dict,
            retrieved_examples: List[dict],
            related_algorithm: List[dict],
            logic_for_this_task: str,
            apis_for_this_task: list[str],
            openai_client : OpenAIClient,
            load_few_shots: bool = True,
        ) -> str:
        """
        Run the editor agent to generate the code.

        parameter:
        -task: A dictionary containing task information.
        -retrieved_examples: The retrieved list of examples, each example is a dictionary.
        -related_algorithm: A list of related algorithms, each algorithm is a dictionary.
        -logic_for_this_task: Logical modeling (algorithm) used for this task.
        -apis_for_this_task: List of APIs used for this task.
        -openai_client : The big model client used to call
        -load_few_shots: Whether to load the few-shot example.

        return:
        -scl_code: The generated code string.
        """

        requirement = str(task)

        # Generate API description based on the recommended API
        if apis_for_this_task:
            api_description = APIDataLoader.format_api_details(apis_for_this_task)
        else:
            api_description = "No Control Instruction is recommended for this task. Please determine the required operations manually."

        # Algorithm generation guidance
        logic_for_this_task = f"<!-- A Control Logic that you can refer to when coding -->\n<ControlLogic>\n{logic_for_this_task}\n</ControlLogic>\n"

        editor_system_prompt = sys_prompt.format(
            api_details=api_description,
            programming_guidance=programming_guidance
        )

        instance_prompt = shot_prompt.format(
            task_requirements=requirement,
            algorithm_process=logic_for_this_task
        )

        code_messages = [{"role": "system", "content": editor_system_prompt}]

        if load_few_shots:
            fewshots = cls.load_code_gen_shots(retrieved_examples, related_algorithm)
            code_messages.extend(fewshots)

        code_messages.append({"role": "user", "content": instance_prompt})

        logger.info("start generation")   
        
        response = openai_client.call(code_messages,task_name=task["name"],role_name="editor")

        raw_content = cls.extract_content(response)

        scl_code = cls.extract_code_from_response(raw_content)

        return scl_code
    
    @classmethod
    def extract_content(cls,response) -> str:
        """
        Extract the SCL code in the response content.

        This method extracts the generated SCL code from the response of the AI ​​model. The response may contain multiple choices,
        Each selection may contain a message object that contains the required content.

        parameter:
        -response: The response object of the AI ​​model.

        return:
        -scl_code: Extracted SCL code string.
        """
        if hasattr(response, 'choices'):
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                scl_code = choice.message.content
            else:
                scl_code = choice['message']['content']
        else:
            scl_code = response.content[0].text
        return scl_code

    @classmethod
    def load_code_gen_shots(self, examples: List[dict], algorithms: List[str]):
        """
        Load and format code generation examples and algorithms into few-shot prompts and add them to the shots list.
        
        Parameters:
        examples (List[dict]): A list of dictionaries, each containing a description and code example.
        algorithms (List[str]): A list of algorithm descriptions corresponding to the examples.
        """
        formatted = []
        for example, algo in zip(examples, algorithms):
            icl_input = shot_prompt.format(
                task_requirements=str(example['json_desc']),
                # algorithm_process = "",
                algorithm_process=f"<! -- A plan that you can refer to when coding -->\n<Plan>\n{algo}\n</Plan>\n"
            )
            icl_output = PromptResultUtil.remove_braces(example['code'])
            formatted.append({"role": "user", "content": icl_input})
            formatted.append({"role": "assistant", "content": icl_output})
        return formatted

sys_prompt = """
Based on the information provided, write SCL code to meet the task requirements. Try to use loops, branches, and sequential structures instead of library functions whenever possible, and only use library functions when necessary. Follow the programming standards. Follow the provided code template to correctly construct SCL code. Refer to the example code of SCL library functions. 

IMPORTANT: If your implementation refers to any auxiliary blocks (e.g., user defined FUNCTION BLOCKs or FUNCTIONs), you must provide complete definitions for all of them. 

Output blocks  **in sequence**, each wrapped as shown:

```scl
FUNCTION_BLOCK <block_name>
    // function block code here
END_FUNCTION_BLOCK
```

```scl
FUNCTION <function_name> : <return_type>
    // function code here
END_FUNCTION
```

SCL Standard Library Documentation:
{api_details}

SCL programming guidances:
{programming_guidance}

""".strip()

programming_guidance = """
1. Do not use SCL syntax that is not allowed in the Siemens S7-1200/1500 PLC.
2. Input and output formats must conform to task requirements.
3. use loops, branches, and sequential structures to achieve objectives.
4. avoid variable name conflicts with keywords and standard function names.
5. define and initialize all loop variables used in FOR loops before use.
6. It is necessary to fully define all the parameters provided in the requirements.
"""

# programming_guidance = "NO PROGRAMMING GUIDANCE PROVIDED"

# programming_guidance = """
# 1. Do not use SCL syntax that is not allowed in the Siemens S7-1200/1500 PLC.
# 2. Input and output formats must conform to task requirements.
# 3. use loops, branches, and sequential structures to achieve objectives.
# 4. avoid variable name conflicts with keywords and standard function names such as `len`.
# 5. define and initialize all loop variables used in FOR loops (e.g., `i`) before use.
# 6. Here is a sample code of state machine programming you may refer to. Output based on oldState when no rising edge/jump is triggered.
# ```scl
# // Assuming it is a rising edge jump
# IF #state AND NOT #oldState:
#     // Update status and operate according to the new status
# END_IF;
# CASE #state OF 
#     #CONST_STATE1: // Operate based on the updated status
#         //do something
#     #CONST_STATE2:
#         //do something
# END_CASE;
# #oldState := #state; // Save current state
# ```
# 7. If state transition is involved, use the syntax of REPEAT-UNTIL combined with CASE-OF to ensure that the output of the current cycle can be updated correctly after the state transition. Because the CASE-OF syntax in SCL does not double check for changes in the case among executing cycles, it is necessary to nest REPEAT-UNTIL in the outer layer to complete the state update. Use the temporary variable 'tempExitStateLoop' to exit from repeat.
# ```scl\nREPEAT\n    tempExitStateLoop := TRUE; // Exit only when no state transition occurs\n    CASE #state OF\n        #CONST_STATE1:\n            // do something\n            // if state change then tempExitStateLoop := FALSE; // Set this variable to False after the state transition to avoid operations that do not execute the new state\n        #CONST_STATE2:\n            // do something\n            // if state change then tempExitStateLoop := FALSE; // Set this variable to False after the state transition to avoid operations that do not execute the new state\n        #CONST_STATE3:\n            // do something\n            // if state change then tempExitStateLoop := FALSE; \n    END_CASE;\nUNTIL(TRUE = #tempExitStateLoop)\nEND_REPEAT;\n```

# 8. It is necessary to fully define all the parameters provided in the requirements.
# """

shot_prompt = """Here is the input, structured in XML format:
<! -- SCL programming task requirements to be completed -->
<Task_Requirements>
{task_requirements}
</Task_Requirements>
{algorithm_process}
"""




# import os
# from typing import Tuple, List
# from autoplc_scl.prompts.editor import (
#     editor_system_prompt_en as sys_prompt,
#     editor_shots_prompt_en as shot_prompt,
#     programming_guidance as guidance
# )
# from autoplc_scl.tools import APIDataLoader, PromptResultUtil
# from autoplc_scl.agents.clients import (
#     autoplc_client_anthropic as client,
#     BASE_MODEL as model
# )

# # class EditorAgent():
#     base_logs_folder: str = None

#     @classmethod
#     def run(cls, 
#             task: dict,
#             retrieved_examples: List[dict],
#             related_algorithm: List[dict],
#             algorithm_for_this_task: str
#         ) -> Tuple[str, int, int]:

#         requirement = str(task)
#         name = task["name"]

#         # constructing prompt
#         api_details_str, _ = APIDataLoader.get_api_details(
#             case_names=[item['name'] for item in retrieved_examples],
#             other_api_names=[], 
#         )

#         editor_system_prompt = sys_prompt.format(
#             api_details = api_details_str,
#             programming_guidance = guidance
#         )
        
#         if algorithm_for_this_task:
#             algorithm_for_this_task = f"<! -- A plan that you can refer to when coding -->\n<Plan>\n{algorithm_for_this_task}\n</Plan>\n"
        
#         editor_instance_prompt = shot_prompt.format(
#             task_requirements = requirement,
#             algorithm_process = algorithm_for_this_task
#         )

#         code_messages = [{"role": "system", "content": editor_system_prompt}]
#         for example, algorithm in list(zip(retrieved_examples, related_algorithm))[::-1]:
#             icl_prompt = shot_prompt.format(
#                 task_requirements = str(example['json_desc']),
#                 algorithm_process = f"<! -- A plan that you can refer to when coding -->\n<Plan>\n{algorithm}\n</Plan>\n",
#             )
#             code_messages.append({"role": "user", "content": icl_prompt})
#             code_messages.append({"role": "assistant", "content":PromptResultUtil.remove_braces(example['code'])})
#         code_messages.append({"role": "user", "content": editor_instance_prompt})

#         # print(code_messages)
#         print("start generation")
#         code_response = client.messages.create(
#             model=model,  
#             max_tokens=16*1024,
#             temperature=0.1,
#             top_p=0.9,
#             messages=code_messages,
#         )

#         scl_code = code_response.choices[0]['message']['content']
#         print(code_response.usage.prompt_tokens)
#         in_, out_ = code_response.usage.input_tokens, code_response.usage.completion_tokens
#         log_path = os.path.join(cls.base_logs_folder, f"{name}/editor.log")
#         with open(log_path, "w", encoding="utf8") as f:
#             code_messages.append({"role": "assistant", "content": scl_code})
#             f.write(PromptResultUtil.message_to_file_str(code_messages))
#             f.write(f"Token Usage:\nInput {in_} + Output {out_} = Total {in_+out_}")

#         return scl_code, code_response.usage.input_tokens,code_response.usage.output_tokens