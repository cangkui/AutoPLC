# import os
import json
from typing import List, Dict, Tuple
from common import ROOTPATH
# import dotenv

# dotenv.load_dotenv()


class APIDataLoader():
    RAG_DATA_DIR = ROOTPATH.joinpath("data/rag_data")
    api_detail_dict: dict = None
    functions_usage: dict = None
    # keywords: dict = None
    api_details_str: str = None

    @classmethod
    def init_load(cls, code_type: str):
        if cls.api_detail_dict is None:
            with open(f"{cls.RAG_DATA_DIR}/{code_type}/{code_type}_instruction_detail_all.json", "r", encoding="utf8") as fp:
                cls.api_detail_dict = json.load(fp)
        if cls.functions_usage is None:
            with open(f"{cls.RAG_DATA_DIR}/{code_type}/{code_type}_functions_usage.json","r", encoding="utf8") as fp:
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
    def extract_apis_from_cases(cls, case_names: List[str]) -> List[str]:
        """
        从案例中提取使用的 API 名称。

        参数:
        - case_names: 案例名称列表
        - functions_usage: 每个案例使用的函数字典

        返回:
        - 提取出的 API 名称列表(去重)
        """
        functions_usage = cls.functions_usage
        extracted_apis = []
        for name in case_names:
            extracted_apis.extend(functions_usage.get(name, []))
        extracted_apis = list(set(extracted_apis))
        return extracted_apis

 
    @classmethod
    def format_api_details(cls, api_names: List[str]) -> Tuple[str, dict]:
        """
        将 API 名称转换为格式化字符串，并返回详细信息字典。

        参数:
        - api_names: API 名称列表
        - api_detail_dict: 每个 API 对应的详细信息字典

        返回:
        - 格式化API字符串
        """
        api_detail_dict = cls.api_detail_dict
        api_details = {}
        data_transform = []

        for api in api_names:
            if api in api_detail_dict:
                api_details[api] = api_detail_dict[api]
            elif "_TO_" in api:
                data_transform.append(api)

        api_details_str = "\n".join(
            f"{i + 1}. {api_detail_dict[api]}" for i, api in enumerate(api_details)
        )

        data_transform_str = "\n".join(
            f"{i + len(api_details) + 1}. {api}() : Convert {src} type to {dst} type"
            for i, api in enumerate(data_transform)
            for src, dst in [api.split('_TO_')]
        )

        return f"{api_details_str}\n{data_transform_str}\n"

