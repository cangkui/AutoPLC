# encoding:utf-8
from __future__ import print_function
import inspect
import time
import os
import inspect

import scriptengine
 


def split_content(content):
    end_var_index = content.rfind("END_VAR")
    
    if end_var_index == -1:
        return content, ""
    
    part1 = content[:end_var_index + len("END_VAR")]        #variable definition
    part2 = content[end_var_index + len("END_VAR"):]        #function block
    return part1, part2

def pre_exec(content):
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.upper().startswith('END_FUNCTION_BLOCK'):
            break 
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def codesys_debug(filename, content):
    content = pre_exec(content)
    print(content)
    project = scriptengine.projects.open(r"D:\autoPLCASEproject\codesysProject\noUI.project")
    project.create_folder("POU")

    p1, p2 = split_content(content)
    declaration_text = p1
    implementation_text = p2

    folder = project.find('POU', recursive = True)[0]
    struktur = folder.create_pou(filename) # DutType.Structure is the default
    struktur.textual_declaration.replace(declaration_text)
    struktur.textual_implementation.replace(implementation_text)

    project.check_all_pool_objects()

    precompile_message = system.get_messages(category="{217bc73e-759b-4a3c-bfa1-991c938a6541}")
    print(precompile_message)
    return precompile_message
    

# codesys_checker(filename, content)
# if __name__ == "__main__": 
def get_file():
    base_name = ""
    content = ""
    temp_dir = r"D:\autoPLCASEproject\gitAutoPLC\AutoPLC\src\autoplc\autoplc_st\tools\tempfile"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    st_files = []
    for entry in os.listdir(temp_dir):
        if entry.lower().endswith('.st'):
            st_files.append(entry)

    if not st_files:
        raise FileNotFoundError("tempfile file not found")
    
    file_name = st_files[0]
    file_path = os.path.join(temp_dir, file_name)
    base_name = os.path.splitext(file_name)[0]
    print(base_name)
    print(file_path)
    if os.path.exists(file_path):
        print("file exist")
    else:
        print("file not exist")

    if os.access(file_path, os.R_OK):
        print("have read")
    else:
        print("can't read")
    with open(file_path, 'r') as f:
        content = f.read()
    return base_name, content

filename, content = get_file()
codesys_debug(filename, content)