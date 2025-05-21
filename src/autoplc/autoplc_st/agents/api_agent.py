from calendar import c
from dataclasses import dataclass
from os import name
import os
from re import S
import sys
from typing import Tuple, List
import json

from flask import config

from common import Config
from autoplc_st.agents.clients import BM25RetrievalInstruction, ClientManager, OpenAIClient, ZhipuAIQAClient

import logging
logger = logging.getLogger("autoplc_st")

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
    推荐的API类。
    Attributes:
        basic_instructions (List[BasicInstruction]): 基本指令列表。
        library_instructions (List[str]): 库指令列表。
    """
    basic_instructions: List[BasicInstruction]
    library_instructions: List[str]

class ApiAgent():
    """
    分别从相似案例、算法描述中推荐
    
    """
    # 定义一个类变量，用于存储基础日志文件夹的路径
    base_logs_folder: str = ""

    
    @classmethod
    def extract_content(cls,response) -> str:
        """
        提取大模型响应的content
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
    def run_gen_dsl(cls,
            task: dict,
            algorithm_for_this_task: str,
            openai_client : OpenAIClient
        ) -> List[dict]:

        # 将任务信息转换为字符串
        requirement = str(task)
    
        # 生成与数据类型相关的指令消息
        dsl_gen_messages = [
            {"role": "system", "content": gen_dsl_system_prompt_zh},
            {"role": "user", "content" : gen_dsl_user_prompt_zh.format(requirement=requirement,algorithm=algorithm_for_this_task)}
        ]

        # 调用OpenAI的API生成DSL（领域特定语言）指令
        dsl_resp = openai_client.call(
            messages = dsl_gen_messages,
            task_name= 'gen_dsl',
            role_name= 'api_agent',
        ).choices[0].message.content
    
        # 从生成的响应中提取JSON格式的DSL列表
        dsl_list = cls.get_json_from_content(dsl_resp)

        # 遍历DSL列表，查询相关的基本指令
        for dsl in dsl_list:
            print(dsl)

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
        运行推荐API接口函数。
    
        根据给定的任务和算法，使用OpenAI和ZhipuAI客户端来推荐合适的API。
    
        参数:
        - task: 包含任务信息的字典。
        - algorithm_for_this_task: 当前任务使用的算法名称。
        - openai_client: OpenAI客户端实例，用于调用OpenAI的API。
        - zhipuai_client: ZhipuAIQAClient实例，用于调用ZhipuAI的API。
    
        返回:
        - Tuple[List[str], List[str]]: 包含推荐的基本指令和库指令的名字元组。
        """
    
        # 初始化基本指令和库指令列表
        basic_instructions =  []
        library_instructions = []

        # TODO: 生成DSL（领域特定语言）指令
        if algorithm_for_this_task :
            pass
            # dsl_list = cls.run_gen_dsl(task,algorithm_for_this_task,openai_client)
        
            # # 遍历DSL列表，查询相关的基本指令
            # for dsl in dsl_list:
            #     basic_instructions.extend(local_api_retriever.query_api_by_type(dsl['涉及的复杂数据类型']))
            #     basic_instructions.extend(local_api_retriever.query_algo_apis(dsl['触发条件'] + "。"  + dsl['操作内容']))

        # TODO: few-shot setting
        if load_few_shots:
            pass

        # 根据算法语义检索
        # TODO: 改为智谱检索
        if algorithm_for_this_task:
            basic_instructions += local_api_retriever.query_algo_apis(algorithm_for_this_task)
    
        # 去除重复的指令
        basic_instructions = list(set(basic_instructions))
        library_instructions = list(set(library_instructions))
    
        # 打印推荐的基本指令和库指令
        logger.info(f"推荐的基本指令：{basic_instructions}")
        logger.info(f"推荐的库指令：{library_instructions}")

        # 返回推荐的API指令实例
        return basic_instructions, library_instructions

if  __name__ == '__main__':

    task = {"title": "FIFO First-In-First-Out Queue", "description": "Write a function block FB to implement the functionality of a First-In-First-Out (FIFO) circular queue, where the maximum length and data type of the queue are variable. The circular queue should support the following operations:\n\n1. Enqueue operation: Add an element to the end of the queue when the queue is not full.\n2. Dequeue operation: Remove an element from the front of the queue when the queue is not empty and return the value of that element.\n3. Check if the queue is empty: Check if there are no elements in the queue.\n4. Check if the queue is full: Check if the queue has reached its maximum capacity.\n5. Get the number of elements in the queue: Return the current number of elements in the queue.\nStatus codes:\n16#0000: Execution of FB without error\n16#8001: The queue is empty\n16#8002: The queue is full", "type": "FUNCTION_BLOCK", "name": "FIFO", "input": [{"name": "enqueue", "type": "Bool", "description": "Enqueue operation, add an element to the end of the queue when the queue is not full"}, {"name": "dequeue", "type": "Bool", "description": "Dequeue operation, remove an element from the front of the queue when the queue is not empty and return the value of that element."}, {"name": "reset", "type": "Bool", "description": "Reset operation, reset head and tail pointers, elementCount output is set to zero, and isEmpty output is set to TRUE."}, {"name": "clear", "type": "Bool", "description": "Clear operation, reset head and tail pointers, the queue will be cleared and initialized with the initial value initialItem. ElementCount output is set to zero, and isEmpty output is set to TRUE."}, {"name": "initialItem", "type": "Variant", "description": "The value used to initialize the queue"}], "output": [{"name": "error", "type": "Bool", "description": "FALSE: No error occurred TRUE: An error occurred during the execution of FB"}, {"name": "status", "type": "Word", "description": "Status code"}, {"name": "elementCount", "type": "DInt", "description": "The number of elements in the queue"}, {"name": "isEmpty", "type": "Bool", "description": "TRUE when the queue is empty"}], "in/out": [{"name": "item", "type": "Variant", "description": "The value used to add to the queue or return from the queue"}, {"name": "buffer", "type": "Variant", "description": "Used as an array for the queue"}], "status_codes": {"16#0000": "No error in execution of FB", "16#8001": "The queue is empty", "16#8002": "The queue is full"}}
    algo_for_task = "初始化与清空逻辑：当reset为TRUE时，重置队列头尾指针，elementCount设为0，isEmpty设为TRUE，保留原buffer内容，设置error为FALSE，status为16#0000；当clear为TRUE时，重置指针与计数器，isEmpty设为TRUE，并使用initialItem填充整个buffer，设置error为FALSE，status为16#0000。2. 入队操作：当enqueue为TRUE时，若队列已满则error为TRUE，status为16#8002，不修改数据；否则将item写入tail位置，tail后移（循环处理），elementCount加1，isEmpty设为FALSE，error为FALSE，status为16#0000。3. 出队操作：当dequeue为TRUE时，若队列为空则error为TRUE，status为16#8001，item不变；否则将head位置元素赋值给item，head后移（循环处理），elementCount减1，若为0则isEmpty为TRUE，error为FALSE，status为16#0000。4. 辅助状态输出：elementCount实时反映队列元素数，isEmpty依据elementCount是否为0判断，status与error一一对应（正常16#0000，空队列出队16#8001，满队列入队16#8002）。"
    test_config = exp_config = Config(config_file="default")
    ClientManager().set_config(test_config)
    openai_client = ClientManager().get_openai_client()
    zhipuai_client = ClientManager().get_zhipuai_client()

    ApiAgent.run_gen_dsl(task=task,algorithm_for_this_task=algo_for_task,openai_client=openai_client)

gen_dsl_system_prompt_zh = """
角色：你是西门子 S7-1200/1500 系列 PLC 系统的专业工程师，精通顺序控制、状态逻辑与数据块管理。

任务：请结合需求中的复杂数据类型，将用户给出的建模流程描述，解析为许多个结构化的 DSL 表达，以便后续进行指令推荐与程序生成。
你的输出应准确表达控制逻辑中的条件与操作，并标注涉及的复杂数据类型（如 IEC_TIMER 、Variant、 Array[*]、 DTL、 String等）。

示例输出格式如下：

```json
[{
    "触发条件": "无",
    "操作内容": "计算数组的长度",
    "涉及的复杂数据类型": ["ARRAY[*]"]
}，
{
    "触发条件": "水位（#WaterLevel）达到设定值（#Number）",
    "操作内容": "启动泵（#pump）并监控运行时间",
    "涉及的复杂数据类型": ["IEC_TIMER"]
}]
```

重要：
- 每个DSL的操作内容尽可能原子化。
- 触发条件和操作内容应尽量简洁、准确，符合PLC工程师风格。
- 仅需要标注操作涉及的复杂数据类型，因为这些类型通常需要特殊的st指令去进行类型判断、读写操作、数据转换。
- 数据类型应基于操作语义与需求中的参数进行合理推断（如遇到计时操作，考虑IEC_TIMER等）。
"""

gen_dsl_user_prompt_zh = """
## st编程需求
{requirement}

## 针对该需求的建模流程:
{algorithm}
""".strip()

recommend_function_system_prompt_zh = """
角色：你是一位精通西门子 S7-1200/1500 系列 的资深PLC系统架构师，负责基于控制流程模型为工程项目推荐可能使用的自定义函数或模块级封装。

任务目标：
请你结合需求，根据建模生成的控制流程（如状态机、顺序控制段）与操作描述，推理出可能适用的自定义库函数。
你的推荐应结合操作语义、数据类型和变量上下文，帮助开发者高效调用已有封装，而非手动实现底层逻辑。

输出格式如下：

```json
[
    {
        "推荐函数": "MyLib_TriggerAlarm",
        "推荐理由": "用于处理温度超标时的报警操作，已经封装了报警逻辑",
    },
    {
        "推荐函数": "MyLib_EventLogger",
        "推荐理由": "用于记录报警事件，兼容温度值和报警信息",
    }
]
```

要求：
- 函数推荐必须基于控制流程模型中涉及的操作内容。
- 每个推荐函数应具有简洁的推荐理由和调用示例。
- 若无法确定函数，推荐类似功能的模块或命名建议。
"""

recommend_function_user_prompt_zh = """
## st编程需求
{requirement}

## 针对该需求的建模流程:
{logic_for_this_task}

## 候选库函数
{library_functions}

""".strip()



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