
import requests
import os
import time
import json

from dataclasses import dataclass,asdict
from typing import List, Optional, Dict


@dataclass
class ErrorMessage:
    error_desc: str
    error_type: str  # "定义区错误" 或 "代码段错误"
    code_window: Optional[str] = None

    def __str__(self):
        return f"[{self.error_type}] {self.error_desc}\n{self.code_window}"

    @classmethod
    def from_dict(cls, data: dict) -> 'ErrorMessage':
        # 保留兼容性（不建议用 path/is_def 创建，但防止旧数据出错）
        return cls(
            error_desc=data['error_desc'],
            error_type=data.get('error_type', '代码段错误'),
            code_window=data.get('code_window')
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
            errors=[ErrorMessage(
                error_desc="编译工具调用失败",
                error_type="系统错误",
                code_window=None
            )]
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
    
    @classmethod
    def extract_code_window(source_code: str, error_info: Dict, window_size: int = 3) -> str:
        lines = source_code.splitlines()
        path = error_info.get("path", 0)
        is_def = error_info.get("isDef", False)

        base_line_idx = 0
        if not is_def:
            # 尝试找到 BEGIN 作为逻辑基准行
            begin_idx = next((i for i, line in enumerate(lines) if "BEGIN" in line.upper()), None)
            if begin_idx is not None:
                base_line_idx = begin_idx
            else:
                # fallback: 找最后一个 END_VAR
                end_var_indices = [i for i, line in enumerate(lines) if "END_VAR" in line.upper()]
                if end_var_indices:
                    base_line_idx = end_var_indices[-1] + 1
                else:
                    base_line_idx = 0  # fallback

        # 推测的错误行位置
        error_line_idx = base_line_idx + path
        start_idx = max(0, error_line_idx - window_size)
        end_idx = min(len(lines), error_line_idx + window_size + 1)

        # 输出窗口，不标记错误行
        result = []
        for i in range(start_idx, end_idx):
            result.append(f"{i + 1:>4}: {lines[i]}")

        return "\n".join(result)
    def scl_syntax_check(self, block_name: str, scl_code: str) -> ResponseData:
        """
        检查SCL代码语法，并输出简化后的错误信息（含类型、窗口）
        """
        t1 = time.time()
        resp = requests.post(
            url="http://192.168.103.152:9000/api/tiaapi/process",
            json={"BlockName": block_name, "Code": scl_code}
        )
        print(f"TIA Compiler Time usage: {(time.time() - t1):.2f}")

        if resp.status_code == 200:
            raw_data = resp.json()
            raw_errors = raw_data.get("Errors", [])

            simplified_errors = []
            for err in raw_errors:
                error_type = "定义区错误" if err.get("is_def", False) else "代码段错误"
                code_window = self.extract_code_window(
                    scl_code,
                    {"path": err["path"], "isDef": err["is_def"]},
                    window_size=3
                )

                simplified_errors.append(ErrorMessage(
                    error_desc=err["error_desc"],
                    error_type=error_type,
                    code_window=code_window
                ))

            return ResponseData(
                success=raw_data.get("Success", False),
                result=raw_data.get("Result"),
                errors=simplified_errors
            )

        else:
            return ResponseData.default_false()
