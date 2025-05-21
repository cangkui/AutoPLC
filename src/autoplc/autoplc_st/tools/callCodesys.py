import subprocess
import os
# -*- coding: utf-8 -*-

bat_path = r"D:\autoPLCASEproject\gitAutoPLC\AutoPLC\src\autoplc\autoplc_st\tools\callCodesys.bat"

test_content = r"""
"""



test_filename =  ""

class State:
    def __init__(self):
        self.filename = ""
        self.content = ""

state = State() 


def run_bat():
    from pathlib import Path
    current_path = Path(__file__).resolve().parent.joinpath("codesys_debugger.py")
    # try:
    result = subprocess.run(
        [bat_path],
        shell=True,
        check=True,
        text=True,
        capture_output=True,
        timeout=3600
    )
    return result

def pre_exec(content):
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.upper().startswith('END_FUNCTION_BLOCK'):
            break  
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def post_exec(content):
    error_lines = []
    lines = content.split('\n')
    for line in lines:
        error_pos = line.find(': Error:')
        if error_pos != -1:
            error_content = line[error_pos + len(': Error:'):].lstrip()
            error_lines.append(error_content)

    return error_lines

def codesys_check(filename, content):
    
    temp_dir = r"D:\autoPLCASEproject\gitAutoPLC\AutoPLC\src\autoplc\autoplc_st\tools\tempfile"

    for item in os.listdir(temp_dir):
        item_path = os.path.join(temp_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # scl_files = []
    # for entry in os.listdir(temp_dir):
    #     if entry.lower().endswith('.st') and os.path.isfile(os.path.join(temp_dir, entry)):
    #         scl_files.append(entry)
    
    # if not scl_files:
    #     raise FileNotFoundError("tempfile file not found")
    # original_file = os.path.join(temp_dir, scl_files[0])
    file_ext = ".st"
    
    new_filename = f"{filename}{file_ext}"
    new_file_path = os.path.join(temp_dir, new_filename)
    with open(new_file_path, 'w', encoding="utf-8") as f:
        f.write(content)

    print(state.filename)
    compile_res = run_bat()
    print(compile_res)
    final_error = post_exec(compile_res.stderr)
    print(final_error)
    return final_error

def get_name_content():
    return test_filename, test_content
# if __name__ == "__main__":
#     codesys_check(test_filename, test_content)


