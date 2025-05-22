from typing import Tuple, List
from autoplc_st.tools.prompt_res_util import PromptResultUtil
from autoplc_st.tools import APIDataLoader
from autoplc_st.agents.clients import ClientManager, OpenAIClient
import logging
logger = logging.getLogger("autoplc_st")


class LogicComposer():
    """
    LogicComposer类用于生成st代码。它提供了从响应中提取代码的功能，并通过运行编辑器代理生成代码。

    Attributes:
        base_logs_folder (str): 用于存储基础日志文件夹的路径。

    Methods:
        extract_code_from_response(text: str) -> str:
            从给定的文本响应中提取st代码。

        run_gen_st(task: dict, retrieved_examples: List[dict], related_algorithm: List[dict], algorithm_for_this_task: str, model: str) -> str:
            运行编辑器代理以生成st代码。该方法使用任务信息、检索到的示例、相关算法和特定任务的算法描述来生成代码。
    """

    # 定义一个类变量，用于存储基础日志文件夹的路径
    base_logs_folder: str = None

    @classmethod
    def extract_code_from_response(cls, text: str) -> str:
        """
        Extract code from response
        ```ST
            <Code>
        ```
        """
        import re

        # 使用正则表达式提取实际的代码内容
        match = re.search(r'```ST\n(.*?)\n```', text, re.DOTALL)

        if match:
            code_content = match.group(1).strip()  # 使用 strip 去除多余的空格和换行
        else:
            code_content = text
            
        return code_content
    
    @classmethod
    def run_gen_st(cls,
            task: dict,
            retrieved_examples: List[dict],
            related_algorithm: List[dict],
            logic_for_this_task: str,
            apis_for_this_task: list[str],
            openai_client : OpenAIClient,
            load_few_shots: bool = True,
        ) -> str:
        """
        运行编辑器代理以生成代码。

        参数:
        - task: 包含任务信息的字典。
        - retrieved_examples: 检索到的示例列表，每个示例是一个字典。
        - related_algorithm: 相关算法的列表，每个算法是一个字典。
        - logic_for_this_task: 用于此任务的逻辑建模（算法）。
        - apis_for_this_task: 用于此任务的API列表。
        - openai_client : 用于调用的大模型客户端
        - load_few_shots: 是否加载few-shot示例。

        返回:
        - st_code: 生成的代码字符串。
        """

        requirement = str(task)

        # 根据建议的API生成API描述
        if apis_for_this_task:
            api_description = APIDataLoader.format_api_details(apis_for_this_task)
        else:
            api_description = "No Control Instruction is recommended for this task. Please determine the required operations manually."

        # 算法生成指导
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

        # 加载Few-Shot示例
        if load_few_shots:
            fewshots = cls.load_code_gen_shots(retrieved_examples, related_algorithm)
            code_messages.extend(fewshots)

        # 添加用户提示信息到消息列表中
        code_messages.append({"role": "user", "content": instance_prompt})

        logger.info("start generation")   
        
        response = openai_client.call(code_messages,task_name=task["name"],role_name="editor")

        raw_content = cls.extract_content(response)

        st_code = cls.extract_code_from_response(raw_content)

        return st_code
    
    @classmethod
    def extract_content(cls,response) -> str:
        """
        提取响应内容中的st代码。

        该方法从AI模型的响应中提取生成的st代码。响应可能包含多个选择，
        每个选择可能包含一个消息对象，该对象包含所需的内容。

        参数:
        - response: AI模型的响应对象。

        返回:
        - st_code: 提取出的st代码字符串。
        """
        if hasattr(response, 'choices'):
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                st_code = choice.message.content
            else:
                st_code = choice['message']['content']
        else:
            st_code = response.content[0].text
        return st_code

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
Based on the information provided, write ST code to meet the task requirements. Try to use loops, branches, and sequential structures instead of library functions whenever possible, and only use library functions when necessary. Follow the programming standards. Follow the provided code template to correctly construct ST code. Refer to the example code of ST library functions. 

OutputFormat:
```ST
// only your ST code is allowed here.
```

ST Standard Library Documentation:
{api_details}

ST programming guidances:
{programming_guidance}

""".strip()

programming_guidance = "NO PROGRAMMING GUIDANCE PROVIDED"

# programming_guidance = """
# 1. Do not use st syntax that is not allowed in the Siemens S7-1200/1500 PLC.
# 2. Input and output formats must conform to task requirements.
# 3. use loops, branches, and sequential structures to achieve objectives.
# 4. avoid variable name conflicts with keywords and standard function names such as `len`.
# 5. define and initialize all loop variables used in FOR loops (e.g., `i`) before use.
# 6. Here is a sample code of state machine programming you may refer to. Output based on oldState when no rising edge/jump is triggered.
# ```st
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
# 7. If state transition is involved, use the syntax of REPEAT-UNTIL combined with CASE-OF to ensure that the output of the current cycle can be updated correctly after the state transition. Because the CASE-OF syntax in st does not double check for changes in the case among executing cycles, it is necessary to nest REPEAT-UNTIL in the outer layer to complete the state update. Use the temporary variable 'tempExitStateLoop' to exit from repeat.
# ```st\nREPEAT\n    tempExitStateLoop := TRUE; // Exit only when no state transition occurs\n    CASE #state OF\n        #CONST_STATE1:\n            // do something\n            // if state change then tempExitStateLoop := FALSE; // Set this variable to False after the state transition to avoid operations that do not execute the new state\n        #CONST_STATE2:\n            // do something\n            // if state change then tempExitStateLoop := FALSE; // Set this variable to False after the state transition to avoid operations that do not execute the new state\n        #CONST_STATE3:\n            // do something\n            // if state change then tempExitStateLoop := FALSE; \n    END_CASE;\nUNTIL(TRUE = #tempExitStateLoop)\nEND_REPEAT;\n```

# 8. It is necessary to fully define all the parameters provided in the requirements.
# """

shot_prompt = """Here is the input, structured in XML format:
<! -- st programming task requirements to be completed -->
<Task_Requirements>
{task_requirements}
</Task_Requirements>
{algorithm_process}
"""




# import os
# from typing import Tuple, List
# from autoplc_st.prompts.editor import (
#     editor_system_prompt_en as sys_prompt,
#     editor_shots_prompt_en as shot_prompt,
#     programming_guidance as guidance
# )
# from autoplc_st.tools import APIDataLoader, PromptResultUtil
# from autoplc_st.agents.clients import (
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

#         st_code = code_response.choices[0]['message']['content']
#         print(code_response.usage.prompt_tokens)
#         in_, out_ = code_response.usage.input_tokens, code_response.usage.completion_tokens
#         log_path = os.path.join(cls.base_logs_folder, f"{name}/editor.log")
#         with open(log_path, "w", encoding="utf8") as f:
#             code_messages.append({"role": "assistant", "content": st_code})
#             f.write(PromptResultUtil.message_to_file_str(code_messages))
#             f.write(f"Token Usage:\nInput {in_} + Output {out_} = Total {in_+out_}")

#         return st_code, code_response.usage.input_tokens,code_response.usage.output_tokens