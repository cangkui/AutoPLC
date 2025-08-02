from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List
import json
from autoplc_scl.tools import APIDataLoader
from common import Config
from autoplc_scl.agents.clients import BM25RetrievalInstruction, ClientManager, OpenAIClient, ZhipuAIQAClient

import logging
logger = logging.getLogger("autoplc_scl")

@dataclass
class Parameter:
    name: str
    type: str
    description: str

@dataclass
class BasicInstruction:
    name: str
    description: str
    parameters: List[Parameter]

@dataclass 
class RecommendedAPIs:
    """
    Recommended API class.
    Attributes:
        basic_instructions (List[BasicInstruction]): Basic instructions list.
        library_instructions (List[str]): Library instructions list.
    """
    basic_instructions: List[BasicInstruction]
    library_instructions: List[str]

class ApiAgent():
    """
    Recommended from similar cases and algorithm descriptions respectively
    
    """
    # The path used to store the underlying log folder
    base_logs_folder: str = ""

    
    @classmethod
    def extract_content(cls,response) -> str:
        """
        Extract the content of the response of the llm
        """
        if hasattr(response, 'choices'):
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
            else:
                content = choice['message']['content']
        else:
            content = response.content[0].text
        return content
    
    @classmethod
    def get_json_from_content(cls,content:str) : 
        try:
            json_start = content.index('```json') + len('```json')
            json_end = content.index('```', json_start)
            json_str = content[json_start:json_end].strip()
            json_data = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error parsing JSON from content: {e}")
            json_data = None
        return json_data
    
    @classmethod
    def extract_complex_type(
        cls,
        task: dict,
        algorithm_for_this_task: str,
        openai_client: OpenAIClient
    ) -> List[str]:
        """
        Calling llm to extract the complex data types involved from the task requirements.
        Output as a list of strings, for example ["ARRAY[*]", "Variant"]
        """
        requirement = str(task)

        messages = [
            {"role": "system", "content": extract_type_system_prompt_zh},
            {"role": "user", "content": extract_type_user_prompt_zh.format(requirement=requirement, algorithm=algorithm_for_this_task)}
        ]

        try:
            response = openai_client.call(
                messages=messages,
                task_name=task.get('name', 'unknown_task'),
                role_name='extract_complex_type',
            )
            content = cls.extract_content(response)
            complex_types = json.loads(str(content))

            if isinstance(complex_types, list):
                logger.info(f"🔍 Extracted complex types: {complex_types}")
                return complex_types
            else:
                logger.warning("⚠️ Output is not a valid string list.")
                return []
            
        except Exception as e:
            logger.error(f"❌ Failed to extract complex types: {e}")
            return []

    @classmethod
    def run_filter_relevant_functions_group(
        cls,
        task: dict,
        algorithm_for_this_task: str,
        functions_json_list: List[dict],
        openai_client
    ) -> List[str]:
        """
        Group calls to filter out the functions that must be used and return the structure:[{name: ..., reason: ...}, ...]
        """
        group_size = 15 # 15 functions are a group
        groups = [functions_json_list[i:i+group_size] for i in range(0, len(functions_json_list), group_size)]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(cls.run_filter_relevant_functions, task, algorithm_for_this_task, group, openai_client)
                for group in groups
            ]
            results = [future.result() for future in futures]

        # Merge all results
        merged_results = []
        for result in results:
            if isinstance(result, list):
                merged_results.extend(result)
            else:
                logger.warning("⚠️ Output is not a valid function list.")

        logger.info(f"✅ Filtered {len(merged_results)} relevant functions after grouping.")
        return merged_results


    @classmethod
    def run_filter_relevant_functions(
        cls,
        task: dict,
        algorithm_for_this_task: str,
        functions_json_list: List[dict],
        openai_client: OpenAIClient
    ) -> List[str]:
        """
        Given a brief description of tasks, algorithms and functions, call llm to filter out the functions that must be used and the recommended reasons.
        Return to the structure:["api1", "api2", ...]
        """
        requirement = str(task)

        messages = [
            {"role": "system", "content": filter_relevant_instructions_system_prompt},
            {"role": "user", "content": filter_relevant_instructions_user_prompt.format(
                requirement=requirement,
                algorithm=algorithm_for_this_task,
                functions_json=json.dumps(functions_json_list, indent=2)
            )}
        ]

        try:
            response = openai_client.call(
                messages=messages,
                task_name=task.get('name', 'unknown_task'),
                role_name='filter_relevant_functions'
            )
            content = cls.extract_content(response)
            filtered = json.loads(content)

            if isinstance(filtered, list):
                logger.info(f"✅ Filtered {len(filtered)} relevant functions.")
                return filtered
            else:
                logger.warning("⚠️ Output is not a valid function list.")
                return []

        except Exception as e:
            logger.error(f"❌ Failed to filter relevant functions: {e}")
            return []

    @classmethod
    def run_recommend_api(cls,
            task: dict,
            algorithm_for_this_task: str,
            openai_client: OpenAIClient,
            zhipuai_client: ZhipuAIQAClient,
            local_api_retriever : BM25RetrievalInstruction,
            load_few_shots: bool = True,
        ) -> Tuple[List[str], List[str]]:
        """
        Run the recommended API interface function.
    
        Based on the given tasks and algorithms, use OpenAI and ZhipuAI clients to recommend appropriate APIs.
    
        parameter:
        -task: A dictionary containing task information.
        -algorithm_for_this_task: The name of the algorithm used by the current task.
        -openai_client: OpenAI client instance, used to call OpenAI's API.
        -zhipuai_client: ZhipuAIQAClient instance, used to call ZhipuAI's API.
    
        return:
        -Tuple[List[str], List[str]]: A tuple containing the recommended basic and library instructions.
        """
    
        # Initialize the list of basic instructions and library instructions
        basic_instructions =  []
        library_instructions = []

        # Deprecated: Generate dsl (domain-specific language) directives
        if algorithm_for_this_task :
            pass

        # Deprecated: few-shot setting
        if load_few_shots:
            pass

        # Basic instructions related to query based on task description and algorithm description
        basic_instructions += local_api_retriever.query_multi_channel(task['description'])
        if algorithm_for_this_task:
            basic_instructions += local_api_retriever.query_multi_channel(algorithm_for_this_task)

        # Recall instructions based on complex types
        complex_types = cls.extract_complex_type(task, algorithm_for_this_task, openai_client)
        logger.info(f"🔍 Extracted complex types: {complex_types}")
        if complex_types:
            basic_instructions.extend(local_api_retriever.query_api_by_type(complex_types))

        # Query API-related information (used to reorder)
        if basic_instructions:
            basic_instruction_list = APIDataLoader.query_api_brief(basic_instructions)

        logger.info(f"🔍 Extracted basic instructions: {basic_instructions}")

        # LLM filtering reordering
        if basic_instructions:
            basic_instructions = cls.run_filter_relevant_functions_group(
                task, 
                algorithm_for_this_task, 
                basic_instruction_list, 
                openai_client
            )


        # Remove duplicate instructions
        basic_instructions = list(set(basic_instructions))
        library_instructions = list(set(library_instructions))
        logger.info(f"Recommended basic instructions:{basic_instructions}")
        logger.info(f"Recommended library directives:{library_instructions}")

        # Return to the recommended API directive instance
        return basic_instructions, library_instructions


knowledgebase_qa_prompt= """
Here is a natural language description of a task requirement.
\"\"\"
{{question}}
\"\"\"
Now retrieve some cases from follows:
\"\"\"
{{knowledge}}
\"\"\"
Please answer.
""".strip()


api_retrival_agent_prompt_sys_en: str = """
# Role
You are a searcher. Given a task, you can retrieve the most relevant structured text programming cases (at least 3 cases) from the knowledge base and analyze it.

## Input
1. Given a description of the task requirements. This is a natural language description of some goal or task that the user wants to achieve by programming in ST.

## Goals
1. Find relevant ST programming cases from the knowledge base and sort them in order of similarity.
2. Describe these cases.
3. Identify how these cases can help in solving the given task.

## Constraints
1. Your answer must follow the specified output format. Note that there are some comments prompting you for what to put in that position.
2. The cases in your answer must come from the search results.
3. Each unique name must correspond to a retrieved case.
4. Do not add additional explanations or text outside of the specified output format.

## Workflow
1. Read and understand the task requirement description. 
2. Retrieve the most relevant programming cases (at least 3 cases) from the knowledge base and sort them. 
3. For each case, describe it.
4. Think about what we can learn from these cases to solve the given task. Describe how these cases can help in solving the given task.
5. Present the final result in the specified output format.

## Output Format
<root>
<case>
# Retrieve relevant cases. Write each case in the following format.
<name>
# A unique name for the case. The name can be obtained from the knowledge base during the retrieval process.
</name>
<description>
# Description of the case.
</description>
<assistance>
# How these cases helped in solving the given task, write down your analysis.
</assistance>
</case>
# Similarly add more cases...
<case> ... </case>
</root>

""".strip()

extract_type_system_prompt_zh = """
你是西门子 S7-1200/1500 系列 PLC 编程专家，擅长从任务描述中识别涉及的复杂数据类型（如 Variant、DTL、ARRAY[*]、STRUCT、STRING 等）。

请你阅读用户的任务目标和控制逻辑设计，并判断是否存在需要使用上述复杂数据类型的情况（如：动态变量、时间戳处理、数组操作等）。

输出格式为 JSON 数组，仅包含推测涉及的复杂数据类型。例如：
["Variant", "ARRAY[*]", "DTL"]

注意事项：
- 只返回涉及的复杂数据类型名称。
- 遇到循环缓冲区、队列等应返回 ARRAY[*]。
- 遇到时间、日期、定时器等应返回 DTL 或 IEC_TIMER。
- 如果无涉及，返回空数组 []。
""".strip()

extract_type_user_prompt_zh = """
## 任务描述
{requirement}

## 控制建模设计
{algorithm}
""".strip()


filter_relevant_instructions_system_prompt = """
You are a senior PLC software engineer. Your job is to help select useful existing instructions necessary for the task.

Each instruction is described by:
- name
- functional_summary
- usage_context

You are also given the task description and its control logic plan.

Apply Occam's razor: select the minimal set of indispensable instructions required to fulfill the task.
- Can the task be accomplished without this instruction?
- Does the instruction type align with the task requirement?	

Respond in a JSON array like this:
["api1","api2"]

IMPORTANT: Do not include any other text or explanation. Just the JSON array.
""".strip()


filter_relevant_instructions_user_prompt = """
## Task Requirement
{requirement}

## Control Logic
{algorithm}

## Candidate Instructions
{functions_json}
""".strip()
