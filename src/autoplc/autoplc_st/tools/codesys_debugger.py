# # encoding:utf-8
# from __future__ import print_function
# import inspect
# import time
# import os
# import inspect

# import scriptengine
 


# def split_content(content):
#     end_var_index = content.rfind("END_VAR")
    
#     if end_var_index == -1:
#         return content, ""
    
#     part1 = content[:end_var_index + len("END_VAR")]        #variable definition
#     part2 = content[end_var_index + len("END_VAR"):]        #function block
#     return part1, part2

# def pre_exec(content):
#     lines = content.split('\n')
#     cleaned_lines = []

#     for line in lines:
#         stripped_line = line.strip()
#         if stripped_line.upper().startswith('END_FUNCTION_BLOCK'):
#             break 
#         cleaned_lines.append(line)
#     return '\n'.join(cleaned_lines)

# def codesys_debug(filename, content):
#     content = pre_exec(content)
#     print(content)
#     project = scriptengine.projects.open(r"D:\autoPLCASEproject\codesysProject\noUI.project")
#     project.create_folder("POU")

#     p1, p2 = split_content(content)
#     declaration_text = p1
#     implementation_text = p2

#     folder = project.find('POU', recursive = True)[0]
#     struktur = folder.create_pou(filename) # DutType.Structure is the default
#     struktur.textual_declaration.replace(declaration_text)
#     struktur.textual_implementation.replace(implementation_text)

#     project.check_all_pool_objects()

#     precompile_message = system.get_messages(category="{217bc73e-759b-4a3c-bfa1-991c938a6541}")
#     print(precompile_message)
#     return precompile_message
    

# # codesys_checker(filename, content)
# # if __name__ == "__main__": 
# def get_file():
#     base_name = ""
#     content = ""
#     temp_dir = r"D:\autoPLCASEproject\gitAutoPLC\AutoPLC\src\autoplc\autoplc_st\tools\tempfile"
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)

#     st_files = []
#     for entry in os.listdir(temp_dir):
#         if entry.lower().endswith('.st'):
#             st_files.append(entry)

#     if not st_files:
#         raise FileNotFoundError("tempfile file not found")
    
#     file_name = st_files[0]
#     file_path = os.path.join(temp_dir, file_name)
#     base_name = os.path.splitext(file_name)[0]
#     print(base_name)
#     print(file_path)
#     if os.path.exists(file_path):
#         print("file exist")
#     else:
#         print("file not exist")

#     if os.access(file_path, os.R_OK):
#         print("have read")
#     else:
#         print("can't read")
#     with open(file_path, 'r') as f:
#         content = f.read()
#     return base_name, content

# filename, content = get_file()
# codesys_debug(filename, content)



import os
import time
import json
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class ErrorMessage:
    error_desc: str
    error_type: str
    code_window: Optional[str] = None

    def to_dict(self):
        return {
            "error_desc": self.error_desc,
            "error_type": self.error_type,
            "code_window": self.code_window
        }

    @classmethod
    def from_dict(cls, data: dict):
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
        return {
            "success": self.success,
            "result": self.result,
            "errors": [e.to_dict() for e in self.errors] if self.errors else []
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict):
        errors = data.get("errors", [])
        return cls(
            success=data["success"],
            result=data.get("result"),
            errors=[ErrorMessage.from_dict(e) for e in errors]
        )

    @classmethod
    def default_false(cls):
        return cls(
            success=False,
            result=None,
            errors=[ErrorMessage(
                error_desc="编译工具调用失败",
                error_type="系统错误"
            )]
        )

class CodesysCompiler:
    def extract_code_window(self, source_code: str, error_info: Dict, window_size: int = 3) -> str:
        lines = source_code.splitlines()
        path = error_info.get("Path", 0)
        is_def = error_info.get("IsDef", False)

        base_line_idx = 0
        if not is_def:
            begin_idx = next((i for i, line in enumerate(lines) if "BEGIN" in line.upper()), None)
            if begin_idx is not None:
                base_line_idx = begin_idx
            else:
                end_var_indices = [i for i, line in enumerate(lines) if "END_VAR" in line.upper()]
                base_line_idx = end_var_indices[-1] + 1 if end_var_indices else 0

        error_line_idx = base_line_idx + path
        start_idx = max(0, error_line_idx - window_size)
        end_idx = min(len(lines), error_line_idx + window_size + 1)

        return "\n".join(f"{i + 1:>4}: {lines[i]}" for i in range(start_idx, end_idx))
    def syntax_check(self, block_name: str, st_code: str) -> ResponseData:
        API_KEY = "admin"  # Default API key, change in production
        # Configure requests session
        session = requests.Session()
        session.headers.update({
            'Authorization': 'ApiKey ' + API_KEY,
            'Content-Type': 'application/json'
        })
        URL = "http://192.168.103.117:9000/api/v1/pou/workflow"
        json_data = {"BlockName": block_name, "Code": st_code}
        timeout = 80  # Set a reasonable timeout for the request
        try:
            resp = session.post(URL, json=json_data, timeout=timeout)  # Reasonable timeout

            print(resp.json())
        
            if resp.status_code != 200:
                return ResponseData.default_false()

            raw_data = resp.json()
            raw_errors = raw_data.get("Errors", [])
            simplified_errors = []

            for err in raw_errors:
                code_window = self.extract_code_window(st_code, err, window_size=3)
                simplified_errors.append(ErrorMessage(
                    error_desc=err["ErrorDesc"],
                    error_type="Declaration Section Error" if err.get("IsDef", False) else "Implementation Section Error",
                    code_window=code_window
                ))

            return ResponseData(
                success=raw_data.get("Success", True),
                result=raw_data.get("Result", ""),
                errors=simplified_errors
            )
        except Exception as e:
            print(f"[Error] Codesys Compiler API failed: {e}")
            return ResponseData.default_false()
