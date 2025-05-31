import requests
import os
import time
import json

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import logging

logger = logging.getLogger("autoplc_scl")

@dataclass
class ErrorMessage:
    error_desc: str
    error_type: str  # "Definition Zone Error" or "Segment Error"
    code_window: Optional[str] = None

    def __str__(self):
        return f"[{self.error_type}] {self.error_desc}\n{self.code_window}"

    @classmethod
    def from_dict(cls, data: dict) -> 'ErrorMessage':
        # Handle missing path or is_def, setting path to -1 for no path found
        path = data.get('Path', -1)
        code_window = None if path == -1 else data.get('code_window')

        return cls(
            error_desc=data['error_desc'],
            error_type=data.get('error_type', '代码段错误'),
            code_window=code_window
        )
    
    def to_dict(self):
        return {
            "error_desc": self.error_desc,
            "error_type": self.error_type,
            "code_window": self.code_window
        }
    
    def __str__(self):
        return f"[{self.error_type}] {self.error_desc}\n{self.code_window}"


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
        return cls(
            success=data['Success'],
            result=data['Result'],
            errors=errors
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
        Test all .scl files in the folder in batches, and summarize statistical information such as the pass rate and the number of errors.

        parameter:
        -folder_path: The folder path containing the .scl file.

        return:
        -summary: A dictionary containing information such as pass rate, number of errors, etc.
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
    
    def extract_code_window(self, source_code: str, error_info: Dict, window_size: int = 3) -> str:
        lines = source_code.splitlines()
        path = error_info.get("Path", -1)
        is_def = error_info.get("IsDef", False)

        base_line_idx = 0
        if not is_def:
            # Try to find BEGIN as the logical baseline
            begin_idx = next((i for i, line in enumerate(lines) if "BEGIN" in line.upper()), None)
            if begin_idx is not None:
                base_line_idx = begin_idx
            else:
                # fallback: Find the last END_VAR
                end_var_indices = [i for i, line in enumerate(lines) if "END_VAR" in line.upper()]
                if end_var_indices:
                    base_line_idx = end_var_indices[-1] + 1
                else:
                    base_line_idx = 0  # fallback

        # If path is -1, it means there is no path information and returns to the empty window
        if path == -1:
            return ""

        # Predicted error line location
        error_line_idx = base_line_idx + path
        start_idx = max(0, error_line_idx - window_size)
        end_idx = min(len(lines), error_line_idx + window_size + 1)

        # Output window, no error line marked
        result = []
        for i in range(start_idx, end_idx):
            result.append(f"{i + 1:>4}: {lines[i]}")

        return "\n".join(result)
    
    def scl_syntax_check(self, block_name: str, scl_code: str) -> ResponseData:
        """
        Check the SCL code syntax and output simplified error information (including type and window)
        """
        t1 = time.time()
        resp = requests.post(
            url="http://192.168.103.152:9000/api/tiaapi/process",
            json={"BlockName": block_name, "Code": scl_code}
        )
        logger.info(f"TIA Compiler Time usage: {(time.time() - t1):.2f}")

        if resp.status_code == 200:
            raw_data = resp.json()
            raw_errors = raw_data.get("Errors", [])
            
            simplified_errors = []
            # Used to record line numbers that have already occurred
            error_lines = set()

            for err in raw_errors:
                error_type = "Declaration Section Error" if err.get("IsDef", False) else "Implementation Section Error"
                code_window = self.extract_code_window(
                    scl_code,
                    {"Path": err["Path"], "IsDef": err["IsDef"]},
                    window_size=3
                )

                # Extract the wrong line number
                path = err.get("Path", -1)
                is_def = err.get("IsDef", False)
                lines = scl_code.splitlines()
                base_line_idx = 0
                if not is_def:
                    begin_idx = next((i for i, line in enumerate(lines) if "BEGIN" in line.upper()), None)
                    if begin_idx is not None:
                        base_line_idx = begin_idx
                    else:
                        end_var_indices = [i for i, line in enumerate(lines) if "END_VAR" in line.upper()]
                        if end_var_indices:
                            base_line_idx = end_var_indices[-1] + 1
                        else:
                            base_line_idx = 0
                error_line_idx = base_line_idx + path

                # Check if the line number already exists in the collection
                if error_line_idx not in error_lines:
                    simplified_errors.append(ErrorMessage(
                        error_desc=err["ErrorDesc"],
                        error_type=error_type,
                        code_window=code_window
                    ))
                    error_lines.add(error_line_idx)

            return ResponseData(
                success=raw_data.get("Success", True),  # If there is no Success field, the default return is True
                result=raw_data.get("Result", ""),  # If there is no Result field, the default return of an empty string
                errors=simplified_errors  # Returns simplified error message
            )

        else:
            return ResponseData.default_false()
