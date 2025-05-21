# import os
import json
from typing import List, Dict, Tuple
from common.config import Config
from common import ROOTPATH
# import dotenv

# dotenv.load_dotenv()


class APIDataLoader():
    api_detail_dict: dict = None
    functions_usage: dict = None
    # keywords: dict = None
    api_details_str = ""

    @classmethod
    def init_load(cls, config: Config):
        if cls.api_detail_dict is None:
            cls.api_detail_dict = {}
            try:
                for path in config.INSTRUCTION_PATH:
                    with open(path, "r", encoding="utf8") as fp:
                        for line in fp:
                            json_data = json.loads(line)
                            api_name = json_data["instruction_name"]
                            cls.api_detail_dict[api_name] = json_data
            except Exception as e:
                print(f"Error loading API data: {e}")
        if cls.functions_usage is None:
            with open(config.FUNCTION_USAGE_PATH,"r", encoding="utf8") as fp:
                cls.functions_usage = json.load(fp)

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
        output += f"- description : {data['generated_brief']['functional_summary']}\n"
        output += f"- how to use: {data['how_to_use']}\n"
        
        for category in data['parameters']:
            output += f"- {category} parameters:\n"
            for param in data['parameters'][category]:
                output += f"-- {param['name']} ({param['type']}): {param['description']}\n"
            output += "\n"
        
        output += f"- example_code:```scl\n {data['example_code']}\n```\n"
        
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
    def query_api_brief(cls, api_names: List[str]) -> List[dict]:
        """
        根据 API 名称查询 API 简短信息。

        参数:
        - api_names: API 名称列表

        返回:
        - 包含 API 名称和简短信息的字典列表
        """
        api_detail_dict = cls.api_detail_dict
        api_details = []
        for api in api_names:
            if api in api_detail_dict:
                api_details.append(
                    {
                        "instruction_name": api,
                        "generated_brief": api_detail_dict[api]["generated_brief"],
                        "generated_keywords": api_detail_dict[api]["generated_keywords"],
                    }
                )
        return api_details
    
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
    
        # to be compatible with the verifier
        cls.api_details_str = api_details_str

        return f"{api_details_str}\n{data_transform_str}\n"

