# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import io

EXPORT_PATH = r"C:\Temp\OSCAT_BASIC_Export"

# lib_file_path = r"D:\...\BASIC.library"
# print("Opening library as a temporary project...")
# if run by CODESYS.exe in cmd
# project = projects.open(lib_file_path)
# directly run in codesys IDE (which requires opening the .library file like opening project first)
project = projects.primary

def export_pou():
    export_count = [0]
    print("--- Start Exporting POUs ---")

    def traverse_and_export(node, path=EXPORT_PATH):
        node_str = str(node)
        if node.has_textual_declaration:
            # it is a pou block
            # output to a file, which needs to get the tree path and create the folder if not exist
            file_name = "{}.st".format(node.get_name())
            file_path = os.path.join(path, file_name)
            if not os.path.exists(path):
                os.makedirs(path)
            
            cont = ""
            if node.has_textual_implementation:
                decl_str = node.textual_declaration.text.strip()
                impl_str = node.textual_implementation.text.strip()
                types = ["FUNCTION_BLOCK", "FUNCTION", "PROGRAM"]
                for tp in types:
                    if decl_str.startswith(tp):
                        cont = "{}\n{}\nEND_{}".format(decl_str, impl_str, tp)
                        break
            else:
                cont = node.textual_declaration.text

            if cont == "":
                return

            with io.open(file_path, "w", encoding="utf-8") as pou_file:
                pou_file.write(cont)
                export_count[0] += 1
                print("Exported: {}".format(file_path))
            return
        elif 'ScriptIecLanguageObjectContainerObject' in node_str and \
            'ScriptExternalFileObjectContainer' in node_str:
            # it is a folder, that can be used to construct the tree path
            for child in node.get_children():
                traverse_and_export(child, os.path.join(path, node.get_name()))
        else:
            return
    
    for obj in project.get_children():
        traverse_and_export(obj)
    print("--- Exported {} POUs ---".format(export_count[0]))

export_pou()
