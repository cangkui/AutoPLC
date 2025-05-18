from dataclasses import dataclass
import os
import traceback
from typing import Any, Tuple, List
from autoplc_scl.tools.prompt_res_util import PromptResultUtil
from common import Config
from autoplc_scl.agents.clients import  ZhipuAIQAClient

knowledge_id = os.getenv("SCL_CASE_KNOWLEDGE")
@dataclass
class RetrievedExample:
    name: str
    json_desc: dict
    code: str

    @staticmethod
    def from_dict(data: dict) -> 'RetrievedExample':
        # print(data)
        return RetrievedExample(
            name=data.get("name", ""),
            json_desc=data,
            code=data.get("code", "")
        )
    
    def __getitem__(self, key: str) -> Any:
        if key.startswith("_") or not hasattr(self, key):
            raise KeyError(f"'{key}' is not a valid public attribute")
        return getattr(self, key)


class Retriever():
    """
    Retriever类用于从知识库中检索与给定任务相关的示例。

    Attributes:
        alternatives (List[str]): 备用案例名称列表。
        base_logs_folder (str): 用于存储基础日志文件夹的路径。

    Methods:
        run_retrieve_case(task: dict) -> Tuple[List[RetrievedExample], int, int]:
            运行检索代理以获取与任务相关的示例。

        get_result(response: str, task_name: str) -> List[dict]:
            解析响应以提取检索到的示例，并进行反作弊处理。
    """

    base_logs_folder = None
    
    @classmethod
    def run_retrieve_case(cls, alternatives:List[str] , task: dict, zhipuai_client:ZhipuAIQAClient) -> List[RetrievedExample]:
        """
        运行检索案例方法，用于从知识库中检索相关信息。
        
        参数:
        - alternatives: 一个字符串列表，包含可能的备选答案。
        - task: 一个字典，包含任务的描述和名称。
        - zhipuai_client: ZhipuAIQAClient 实例，用于调用知识库查询。
    
        返回:
        - 一个 RetrievedExample 实例列表，包含检索到的示例。
        """
        
        # 构建消息列表，包括系统和用户角色的消息
        messages=[
            {"role": "system", "content": retrieval_agent_prompt_sys_en},
            {"role": "user", "content": task["description"]}
        ]        
    
        # 调用知识库查询，获取响应
        response = zhipuai_client.call_kbq(messages, 
                                       task_name=task["name"], 
                                       role_name="retriever",
                                       qa_prompt=knowledgebase_qa_prompt,
                                       knowledge_id=knowledge_id)
        
        # 调用 get_result 方法处理响应内容，返回结果列表
        return cls.get_result(alternatives,response.choices[0].message.content, task["name"])


    @classmethod
    def get_result(cls, alternatives:List[str] , response: str, task_name: str) -> List[dict]:
        """
        解析响应以提取检索到的示例，并进行反作弊处理。

        参数:
        - response: 从知识库检索到的响应字符串。
        - task_name: 当前任务的名称。

        返回:
        - res: 包含检索到的示例的列表，每个示例是一个字典。
        """
        res = []
        try:
            response = PromptResultUtil.extract_xml_content(response)
            response = PromptResultUtil.replace_tag(response, "name")
            response = PromptResultUtil.replace_tag(response, "description")
            response = PromptResultUtil.replace_tag(response, "assistance")
            response = PromptResultUtil.parse_xml(response)

            if isinstance(response["case"], dict):
                response["case"] = [response["case"]]

            # anti-cheating
            names = [example["name"].strip() for example in response["case"]]
            names = [name for name in names 
                     if not (name.lower() in task_name.lower() or task_name.lower() in name.lower())]
            names = names + [a for a in alternatives if a not in names]
            
            for name in names[:3]:
                json_desc = PromptResultUtil.get_json_content(name=name, code_type="scl")
                if json_desc is None:
                    continue
                code_scl = PromptResultUtil.get_source_code(name=name, code_type="scl")
                res.append(
                    RetrievedExample.from_dict({
                        "name": name,
                        "json_desc": json_desc,
                        "code": code_scl
                    }))
        except Exception as e:
            print("Error in retriever agent:", e)
            traceback.print_exc()
        
        return res


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


retrieval_agent_prompt_sys_en: str = """
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