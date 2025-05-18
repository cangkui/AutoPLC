
import requests
import os
import time
import json

from dataclasses import dataclass,asdict
from typing import List, Optional

@dataclass
class ErrorMessage:
    path: int
    error_desc: str
    is_def: bool

    def __str__(self):
        return f"Path: {self.path}, Error Desc: {self.error_desc}, Is Def Error: {self.is_def}"
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ErrorMessage':
        return cls(
            path=data['path'],
            error_desc=data['error_desc'],
            is_def=data['is_def']
        )

@dataclass
class ResponseData:
    success: bool
    result: Optional[str] = None
    errors: List[ErrorMessage] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else asdict(o))

    @classmethod
    def from_dict(cls, data: dict) -> 'ResponseData':
        errors = data.get('errors')
        if errors:
            errors = [ErrorMessage.from_dict(e) for e in errors]
        # print(data)
        return cls(
            success=data['Success'],
            result=data['Result'],
            errors=data['Errors']
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ResponseData':
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def default_false(cls) -> 'ResponseData':
        return cls(
            success=False,
            result=None,
            errors=[ErrorMessage(path=-1,error_desc="编译工具调用失败",is_def=False)]
        )        


class TIAPortalCompiler():
    
    def batch_test_scl_files(self, folder_path: str) -> dict:
        """
        批量测试文件夹下的所有.scl文件，并总结通过率和错误数量等统计信息。

        参数:
        - folder_path: 包含.scl文件的文件夹路径。

        返回:
        - summary: 包含通过率、错误数量等信息的字典。
        """
        import os

        total_files = 0
        passed_files = 0
        total_errors = 0
        error_details = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.scl'):
                    total_files += 1
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        scl_code = f.read()

                    response = self.scl_syntax_check(file, scl_code)
                    if response.success:
                        passed_files += 1
                    else:
                        total_errors += len(response.errors)
                        error_details.append({
                            'file': file,
                            'errors': response.errors
                        })

        pass_rate = (passed_files / total_files) * 100 if total_files > 0 else 0

        summary = {
            'total_files': total_files,
            'passed_files': passed_files,
            'pass_rate': pass_rate,
            'total_errors': total_errors,
            'error_details': error_details
        }

        return summary

    def scl_syntax_check(self, block_name: str, scl_code: str) -> ResponseData:
        """
        检查给定的SCL代码块的语法。

        该方法通过向本地API发送POST请求来验证SCL代码的语法。请求包含代码块的名称和实际的SCL代码。
        如果请求成功，返回一个包含语法检查结果的ResponseData对象；否则，返回一个默认的失败ResponseData对象。

        参数:
        - block_name: SCL代码块的名称。
        - scl_code: 需要检查的SCL代码字符串。

        返回:
        - ResponseData: 包含语法检查结果的对象，指示代码是否通过语法检查以及任何错误信息。
        """

        t1 = time.time()
        resp = requests.post(
            url="http://localhost:9000/api/tiaapi/process",
            json={"BlockName": block_name, "Code": scl_code}
        )
        print(f"Time usage: {(time.time() - t1):.2f}")

        if resp.status_code == 200:
            res_data = ResponseData.from_dict(resp.json())
            return res_data
        else:
            return ResponseData.default_false()