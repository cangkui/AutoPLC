import xml.etree.ElementTree as ET
import re
import os
import json
from typing import List
from common import ROOTPATH
import logging
logger = logging.getLogger("autoplc_scl")

class PromptResultUtil():
    RAG_DATA_DIR = ROOTPATH.joinpath("data/rag_data")

    @classmethod
    def xml_to_dict(cls, element: ET.Element) -> dict:
        result = {}
        for child in element:
            if child:
                child_data = cls.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    @classmethod
    def parse_xml(cls, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            root = ET.fromstring(response)
        except:
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
            except:
                root = ET.fromstring('<root>\n' + response)
        return cls.xml_to_dict(root)
    
    @classmethod
    def trim_text(cls, text: str, trimmed_text: str) -> str:
        return text.replace(trimmed_text, '').strip()

    @classmethod
    def replace_tag(cls, text: str, tag: str) -> str:
        return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    @classmethod
    def extract_xml_content(cls, text: str) -> str:
        pattern = re.compile(r'<root>[\s\S]*?</root>')
        match = pattern.search(text)
        if match:
            return match.group(0)
        else:
            return ""
    
    @classmethod
    def get_json_content(cls, name: str, code_type: str) -> dict:
        json_path = f"{cls.RAG_DATA_DIR}/{code_type}/{code_type}_case_requirement/{name}.json"
        if not os.path.exists(json_path):
            logging.warning(f"Warning: There is no case json named '{name}' in: {json_path}.")
            return None
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            code_json = f.read()
        return json.loads(code_json)

    @classmethod
    def get_source_code(cls, name: str, code_type: str) -> str:
        code_path = f"{cls.RAG_DATA_DIR}/{code_type}/{code_type}_case_code/{name}.{code_type}"
        if not os.path.exists(code_path):
            logger.warning(f"Warning: There is no case code named '{name}' in: {code_path}.")
            return None
        with open(code_path, 'r', encoding='utf8') as f:
            case_code = f.read()
        return case_code
    
    @classmethod
    def get_plan(cls, name: str, code_type: str) -> str:
        plan_path = f"{cls.RAG_DATA_DIR}/{code_type}/{code_type}_case_plan/{name}.plan"
        if not os.path.exists(plan_path):
            logger.warning(f"Warning: There is no case plan named '{name}' in: {plan_path}.")
            return None
        with open(plan_path, 'r', encoding='utf8') as f:
            case_plan = f.read()
        return case_plan
    
    @classmethod
    def get_plan_diff(cls, name: str, plan_version: str) -> str:
        plan_path = f"{cls.RAG_DATA_DIR}/scl/scl_case_plan/{plan_version}/{name}.plan"
        if not os.path.exists(plan_path):
            logger.warning(f"Warning: There is no case plan named '{name}' in: {plan_path}.")
            return None
        with open(plan_path, 'r', encoding='utf8') as f:
            case_plan = f.read()
        # [兼容性考虑]尝试加载case_plan为json并读取其中的planning字段
        try:
            tmp_case_plan = json.loads(case_plan)["planning"]
            case_plan = tmp_case_plan
        except:
            pass
        return case_plan

    @classmethod
    def remove_braces(cls, code: str) -> str:
        code = code.replace("{ S7_Optimized_Access := 'TRUE' }","HELLOWORLD")
        code = re.sub(r'\{.*?\}', '', code)
        code = code.replace("HELLOWORLD","{ S7_Optimized_Access := 'TRUE' }")
        pattern = re.compile(r"\(\*.*?\*\)", re.DOTALL)
        code = re.sub(pattern, '', code)
        return code    

    @classmethod
    def message_to_file_str(cls, messages: List[dict]) -> str:
        message_str = "*"*10 + " Message BEGIN " + "*"*10 + "\n"
        for message in messages:
            role = message["role"]
            content = message["content"]
            message_str += "#"*10 + f" {role} " + "#"*10 + "\n"
            message_str += f"{content}\n"
        message_str += "*"*10 + f" Message END " + "*"*10 + "\n\n\n"
        return message_str
    
    