import os
import json
from typing import List, Dict, Tuple
import dotenv
from common import ROOTPATH

dotenv.load_dotenv()

RAG_DATA_DIR = os.path.join(ROOTPATH, "data/rag_data")

class APIDataLoader():
    api_detail_dict: dict = None
    functions_usage: dict = None
    # keywords: dict = None
    api_details_str: str = None

    @classmethod
    def init_load(cls, code_type: str):
        if cls.api_detail_dict is None:
            with open(f"{RAG_DATA_DIR}/{code_type}/{code_type}_instruction_detail_all.json", "r", encoding="utf8") as fp:
                cls.api_detail_dict = json.load(fp)
        if cls.functions_usage is None:
            with open(f"{RAG_DATA_DIR}/{code_type}/{code_type}_functions_usage.json","r") as fp:
                cls.functions_usage = json.load(fp)
        # if cls.keywords is None:
        #     with open(f"{RAG_DATA_DIR}/Keywords.json","r") as fp:
        #         cls.keywords = json.load(fp)

    
    @classmethod
    def prettify_api(cls, data: dict) -> str:
        """
        该方法用于将API数据字典格式化为可读的字符串格式。

        参数:
        - data (dict): 包含API详细信息的字典。

        返回:
        - str: 格式化后的API信息字符串，包含指令名称、描述、参数和示例代码等信息。
        """
        output = f"{data['instruction_name']}()\n"
        output += f"- description: {data['brief_description']}\n"
        
        for category in data['parameters']:
            output += f"- {category} parameters:\n"
            for param in data['parameters'][category]:
                output += f"-- {param['name']} ({param['type']}): {param['description']}\n"
            output += "\n"
        
        output += f"- example_code:```scl\n {data['example_code']}\n```\n"
        output += f"- additional_info: {data['additional_info']}\n"
        
        return output

    @classmethod
    def get_api_details(cls, case_names: List[str], api_names: List[str] = []) -> Tuple[str, dict]:
        """
        获取指定用例和API名称的详细信息。

        参数:
        - case_names (List[str]): 用例名称列表。
        - api_names (List[str], optional): API名称列表，默认为空列表。

        返回:
        - Tuple[str, dict]: 返回一个元组，其中包含格式化后的API详细信息字符串和API详细信息字典。
        """
        api_details = {}
        data_transform = []
        functions_usage = cls.functions_usage
        api_detail_dict = cls.api_detail_dict

        # 从案例中读取使用的api
        for name in case_names:
            case_used_functions = functions_usage.get(name, [])
            for api in case_used_functions:
                if api in api_detail_dict:
                    api_details[api] = api_detail_dict[api]
                elif "_TO_" in api:
                    data_transform.append(api)

        # 直接读取API
        for api in api_names:
            if api in api_detail_dict:
                api_details[api] = api_detail_dict[api]

        # 格式化API详细信息为字符串
        api_details_str = "\n".join(
            f"{i + 1}. {cls.prettify_api(api)}" for i, api in enumerate(api_details.values())
        )

        # 格式化数据转换信息为字符串
        data_transform_str = "\n".join(
            f"{i + len(api_details) + 1}. {data_api}() : Convert {data_from} type to {data_to} type"
            for i, data_api in enumerate(data_transform)
            for data_from, data_to in [data_api.split('_TO_')]
        )

        # 合并字符串
        cls.api_details_str = f"{api_details_str}\n{data_transform_str}\n"

        return cls.api_details_str, api_details
