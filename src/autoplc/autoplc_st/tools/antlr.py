import json
import re
import os
from typing import Tuple, List
from antlr4 import *
from antlr4.error.DiagnosticErrorListener import DiagnosticErrorListener
from antlr4.error.ErrorListener import ErrorListener
from .antlr_scl.sclLexer import sclLexer
from .antlr_scl.sclParser import sclParser
from .antlr_scl.myListener import MyListener as MySCLListener

class SCLGrammarErrorListener(ErrorListener):
    def __init__(self, text_lines: List[str]):
        super().__init__()  
        self.text_lines = text_lines
        self.error_log = ''
        self.SUCCESS = True
        self.errors = {}
        self.count = 1
        self.UTILS_DIR = os.path.join(os.getenv("ROOTPATH"), "src/autoplc/utils")
        with open(f"{self.UTILS_DIR}/antlr_scl/g4_grammar.json", "r") as fp:
            self.g4_grammar = json.load(fp)
    
    def syntaxError(
            self, 
            recognizer, 
            offendingSymbol, 
            line, 
            column, 
            msg, 
            e
        ):
        if line not in self.errors :
            self.SUCCESS = False
            context = '\n'.join(self.text_lines[line-5:line+5])
            actual_token = offendingSymbol.text
            rule_stack = recognizer.getRuleInvocationStack()
            final_rule_grammars = self.g4_grammar.get(rule_stack[0],"no related rules")
            msg = f"The syntax checker terminated unexpectedly at `{actual_token}`"
            error_message = (
                f"\n---Syntax Error No.{self.count}---\n"
                f"-Feedback: {msg}\n"
                f"-Context: \n```scl\n{context}\n```\n"
                f"-SCL syntax rules that the code may violate (.g4 format): {final_rule_grammars}" 
            )
            self.error_log += error_message
            self.errors[line] = error_message
            self.count += 1


class CompilerFrontend():
    RAG_DATA_DIR = os.path.join(os.getenv("ROOTPATH"), "data/rag_data")
    UTILS_DIR = os.path.join(os.getenv("ROOTPATH"), "src/autoplc/utils")

    def __init__(self, code_type: str):
        with open(f"{self.RAG_DATA_DIR}/{code_type}/{code_type}_instruction_detail_all.json", "r", encoding='utf-8') as fp:
            all_funcs: dict = json.load(fp)
            self.all_funcs = {key.lower(): value for key, value in all_funcs.items()}
        
        # with open(f"{self.UTILS_DIR}/antlr_{code_type}/g4_grammar.json", "r") as fp:
        #     self.g4_grammar = json.load(fp)
        
    def remove_double_slash_comments(self, code: str) -> str:
        code = re.sub(r'//.*\n', '\n', code)
        return code

    def extract_scl_functions_from_block(self, block_content: str) -> List[str]:
        """
        该函数用于从给定的SCL代码块中提取所有的函数名称。

        它通过递归地解析代码块中的函数调用，识别出所有的函数名称，并返回一个不重复的函数名称列表。

        返回的函数列表中会去除名为'LGF'的函数，因为它被认为是一个特殊的标识符。

        Returns:
            List[str]: 提取出的函数名称列表。
        """
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\('    
        functions = []
        
        def recursive_extract(text):
            matches = re.finditer(pattern, text)
            for match in matches:
                functions.append(match.group(1))
                start = match.end()
                depth = 1
                for i in range(start, len(text)):
                    if text[i] == '(':
                        depth += 1
                    elif text[i] == ')':
                        depth -= 1
                        if depth == 0:
                            inner_text = text[start:i]
                            recursive_extract(inner_text)
                            break
        recursive_extract(block_content)
        ret = set(functions)
        ret.discard('LGF') 
        return list(ret)


    def scl_syntax_check(self, scl_code: str) -> Tuple[str, bool]:
        """
        检查给定的SCL代码的语法。

        解析SCL代码，识别所有函数调用，并验证这些函数是否在已知函数列表中。
        如果发现未知函数调用，返回错误日志，指出这些未知函数的名称和上下文。

        返回:
            Tuple: 包含错误日志和布尔值的元组。
            - 错误日志: 字符串，包含所有语法错误的详细信息。
            - 布尔值: 如果没有发现错误则为True，否则为False。
        """
        error_log = []
        success = True

        # 初始化解析器和监听器
        input_stream = InputStream(scl_code)
        lexer = sclLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = sclParser(stream)
        parser.removeErrorListeners()
        error_logger = SCLGrammarErrorListener(scl_code.split("\n"))
        parser.addErrorListener(error_logger)
        tree = parser.r()

        # 解析树并提取函数
        listener_pack = {}
        listener = MySCLListener(listener_pack)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        function_list = self.extract_scl_functions_from_block(self.remove_double_slash_comments(scl_code))

        # 检查函数有效性
        for func in function_list:
            if self._is_valid_function(func, listener_pack):
                continue
            success = False
            error_log.append(self._generate_error_message(func, error_logger))

        # 返回结果
        combined_error_log = ''.join(error_log)
        return listener_pack['error_log'] + combined_error_log + error_logger.error_log, not listener_pack['has_error'] and success and error_logger.SUCCESS

    def _is_valid_function(self, func: str, listener_pack: dict) -> bool:
        """检查函数是否有效"""
        if func in listener_pack.get('special_type_variable', []):
            return True
        if "_TO_" in func:
            data_from, data_to = func.split("_TO_")
            if data_from != data_to:
                return True
        return func.lower() in self.all_funcs

    def _generate_error_message(self, func: str, error_logger: SCLGrammarErrorListener) -> str:
        """生成错误信息"""
        error_message = (
            f"\n---Error No.{error_logger.count} ---\n"
            f"-Unknown SCL FUNCTION : {func}() ! \n"
            f"-Context : {func}() ! \n"
            f"-Feedback : 1. Carefully analyze the purpose of using the function and replace it with a suitable SCL library function. \n"
            f" 2. Do not define your own function if it is not available in the library function list. Use plain loops, branches, and sequential structures to achieve the goal. \n"
            f" 3. Use canonical type conversion functions\n"
        )
        error_logger.count += 1
        return error_message
