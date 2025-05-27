import os
import re
import json
import time
from typing import Tuple, List
from common import Config
from autoplc_st.tools import APIDataLoader
from autoplc_st.agents.clients import OpenAIClient
import logging
from autoplc_st.tools import CodesysCompiler
logger = logging.getLogger("autoplc_st")

class AutoDebugger():
    """
    AutoDebugger类

    该类负责验证和改进st代码。它通过编译器尝试编译给定的st代码，并在发现错误时使用AI生成的补丁进行修复。
    这个过程会重复进行，直到代码成功编译或达到最大尝试次数。

    类变量:
    - base_logs_folder: str
        用于存储基础日志文件夹的路径。

    方法:
    - run_debugger_with_compiler(cls, task: dict, st_code: str) -> Tuple[str, int, int]
        执行验证和改进过程。接收一个任务字典和一段st代码，尝试编译并验证代码。
        如果代码中存在错误，将尝试通过AI生成的补丁来修复这些错误。
        返回修复后的st代码以及输入输出的token消耗。
    """

    base_logs_folder: str = None

    @classmethod
    def run_debugger_with_compiler(
        cls, 
        task: dict,
        st_code: str,
        max_verify_count: int,
        openai_client: OpenAIClient,
        load_few_shots: bool = True,
    ) -> str:
        """
        执行验证和改进过程。
    
        该方法接收一个任务字典和一段st代码，然后尝试编译并验证代码。
        如果代码中存在错误，它将尝试通过AI生成的补丁来修复这些错误。
        这个过程会重复进行，直到代码成功编译或达到最大尝试次数。
    
        参数:
        - task: 包含任务信息的字典，如任务名称。
        - st_code: 需要编译和验证的st代码。
        - max_verify_count : 最大验证数量
        - openai_client : 用于调用的大模型客户端
        - load_few_shots : 是否加载few-shot

        返回:
        - st_code : 修复后的st代码
        """
    
        # 定义日志文件路径
        syntax_output_file = os.path.join(cls.base_logs_folder, f"{task['name']}/syntax.log")
        verify_information_log = os.path.join(cls.base_logs_folder, f"{task['name']}/verify_info.jsonl")

        logger.info(f"syntax_output_file is > {syntax_output_file}")
        logger.info(f"verify_information_log is > {verify_information_log}")

        # 初始化验证计数器和编译器实例
        verify_count = 0
        compiler = CodesysCompiler()
        start_time = time.time()
        
        # 构建验证器的系统提示信息
        verifier_system_prompt_with_data = verifier_system_prompt.format(
            api_details = APIDataLoader.api_details_str,
            programming_guidance = programming_guidance
        )

        verifier_messages = [{"role": "system", "content": verifier_system_prompt_with_data}]

        # TODO：few-shot未与openness对齐，暂时注释
        if load_few_shots:
            # verifier_fewshots = cls.load_debug_shots()
            # verifier_messages.extend(verifier_fewshots)
            pass

        # 尝试验证和修复代码，直到成功或达到最大尝试次数
        while verify_count < max_verify_count:
            debugging_process_data = {
                "st_before_fix": "",
                "compiler":[],
                "assistant":""
            }
            verify_count += 1

            check_result = compiler.syntax_check(task['name'], st_code)
            no_error = check_result.success
            
            if no_error:
                logger.info(f"{task['name']} SUCCESS!")
                break
            else:
                error_list = []
                
                # 首选 Data Section 的错误，因为Data Section 的错误会引发级联错误
                for error in check_result.errors:
                    if error.error_type == "Declaration Section Error":
                        logger.info(f"Declaration Error >>> {str(error)}")
                        error_list.append(error.to_dict())

                # 如果没有Data Section 的错误，则检查 Program Section 的错误
                if not error_list:
                    for error in check_result.errors:
                        if error.error_type == "Implementation Section Error":
                            logger.info(f'Implementation Error >>>> {str(error)}')
                            error_list.append(error.to_dict())

                error_log = '\n'.join([str(err) for err in error_list])
                logger.info(f'{task["name"]} Start Verification!')

                # 记录所有错误信息
                debugging_process_data["compiler"] = [error.to_dict() for error in check_result.errors]
            
            verifier_instance_prompt_with_data = verifier_instance_prompt.format(
                static_analysis_results = error_log,
                st_code = st_code
            )
            verifier_messages.append({"role": "user", "content": verifier_instance_prompt_with_data})
            debugging_process_data["st_before_fix"] = st_code

            # 使用AI模型生成修复补丁
            response = openai_client.call(
                messages=verifier_messages,
                task_name=task['name'],
                role_name="verifier"
            )

            verify_result = cls.extract_content(response)
    
            verifier_messages.append({"role": "assistant", "content": verify_result})
            debugging_process_data["assistant"] = verify_result
            segments_and_patches = cls.parse_patch(verify_result)
    
            # 应用修复补丁到代码中
            for buggy, patch in segments_and_patches:
                st_code = st_code.replace(buggy, patch)
            
            with open(verify_information_log, "a+", encoding="utf-8") as fp:
                json.dump(debugging_process_data, fp, ensure_ascii=False)
                fp.write('\n')
                    
            logger.info(f"{task['name']} End-Verify_and_improver ({verify_count}) -Execution time: {(time.time() - start_time):.6f} Seconds")
        
        # 将最终的代码结果写入文件 为方便读取，直接覆盖前面写的内容
        code_output_file = os.path.join(cls.base_logs_folder, f"{task['name']}/{task['name']}_{verify_count}.st")
        logger.info(f"output file is {code_output_file}")
        with open(code_output_file, "w", encoding="utf-8") as fp:
            fp.write(st_code)

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
    def parse_patch(cls, text: str) -> List[str]:
        """
        解析给定文本中的代码段和补丁。
    
        该方法使用正则表达式来查找和提取文本中符合条件的代码段和补丁对。
        它主要关注的是被<code_segment>和<patch>标签包围的文本内容，并将其作为一对 buggy 代码和对应的 patch 提取出来。
    
        参数:
        - text: str - 需要解析的文本。
    
        返回:
        - List[str] - 一个包含多个元组的列表，每个元组包含两个字符串：
          1. buggy 代码（修复前的代码）。
          2. patch（修复该问题的补丁）。
        """
        # 编译一个正则表达式模式，用于匹配<code_segment>和<patch>标签之间的内容。
        # 使用re.DOTALL标志，使.匹配包括换行符在内的所有字符。
        pattern = re.compile(
            r'<code_segment>(.*?)</code_segment>\s*<patch>(.*?)</patch>', 
            re.DOTALL
        )
    
        # 使用编译的正则表达式查找所有匹配项。
        matches: List[str] = pattern.findall(text)
        results = []
        for buggy, patch in matches:
            # 将匹配到的buggy代码和patch去除前后空格后，作为元组添加到结果列表中。
            results.append((buggy.strip(), patch.strip()))
    
        # 返回处理后的结果列表。
        return results
    
    @classmethod
    def load_debug_shots(cls, SHOT_DATA_DIR:str):
        """
        [Deprecated] Load and format code verification examples and algorithms into few-shot prompts and add them to the shots list.
        
        parameters:
        - SHOT_DATA_DIR (str): The directory path where the shot data is located.
        """
        SHOT_DATA_DIR = Config.SHOT_DATA_DIR
        shot_prompt = verifier_instance_prompt
        verifier_shots= []
        
        # 读取验证器的Few-Shot示例
        with open(f'{SHOT_DATA_DIR}/verifier_fewshot.txt','r') as fp:
            verifier_fewshots = fp.read()
    
        verifier_fewshots = verifier_fewshots.split("=====")
        for example in verifier_fewshots:
            buggy_code, check_feedback, response_patch = example.split("&&&&&")
            verifier_icl = shot_prompt.format(
                static_analysis_results = check_feedback,
                st_code = buggy_code
            )
            verifier_shots.append({"role": "user", "content": verifier_icl})
            verifier_shots.append({"role": "assistant", "content": response_patch})
    
        return verifier_shots


verifier_system_prompt = """
ROLE: 
You are an expert in Structured Text (ST) programming for IEC 61131-3-compliant PLCs using the Codesys development environment. 
You specialize in identifying and correcting syntax and semantic errors based on compiler feedback generated by Codesys.

GOALS: 
Generate accurate patch fixes that are syntactically and semantically correct within the Codesys environment.

WORKFLOW:
1. Analyze the compiler error messages provided by the Codesys IDE and locate the actual erroneous ST code segment.
2. Explain why the error occurred using IEC 61131-3 rules and Codesys-specific language semantics.
3. Provide detailed and actionable corrective suggestions for each identified issue.
4. Output a patch using the following required format.

IMPORTANT:
- Assume compilation is performed within the Codesys environment, targeting platforms such as Beckhoff, WAGO, or Schneider Electric PLCs.
- <code_segment> must be a verbatim copy of the original ST code, including indentation and comments.
- <patch> must be a direct replacement for the code segment and must be syntactically valid for Structured Text in Codesys.
- Clearly identify violated syntax rules, common function block misuse, or incorrect data typing.
- Consider that some compiler errors may cascade—reason about task execution flow and variable scope to identify the root issue.
- Avoid altering the original control logic or behavior unless absolutely necessary for correctness.

OUTPUT FORMAT:
```plaintext
- Fix suggestion 1: [Clear and actionable error fix suggestion]
- Fix suggestion k: [Clear and actionable error fix suggestion]
(1)
<code_segment>
# code here
</code_segment>
<patch>
# your patch here
</patch>
(n)
<code_segment>
# code here
</code_segment>
<patch>
# your patch here
</patch>
```

--Appendix--
ST Library Functions:
{api_details}
ST Programming Guidelines:
{programming_guidance}
"""

verifier_instance_prompt = """
Input structured in XML format:
<!-- Code with errors. -->
<STCode>
{st_code}
</STCode>
<!-- Errors. -->
<Errors>
{static_analysis_results}
</Errors>
"""

programming_guidance = "NO PROGRAMMING GUIDANCE"
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

