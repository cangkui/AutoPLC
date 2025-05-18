import re

def analyze_st_code(st_code):

    code_lines = st_code.strip().split('\n')
    code_length = len(code_lines)


    input_vars = re.findall(r'VAR_INPUT', st_code)
    output_vars = re.findall(r'VAR_OUTPUT', st_code)
    input_var_count = len(input_vars)
    output_var_count = len(output_vars)


    loops = re.findall(r'\b(FOR|WHILE|REPEAT)\b', st_code)
    loop_count = len(loops)


    conditions = re.findall(r'\b(IF|CASE)\b', st_code)
    condition_count = len(conditions)


    library_functions = re.findall(r'\b[A-Z_]+\b\(', st_code)
    library_function_count = len(library_functions)

    cyclomatic_complexity = condition_count + 1


    comments = re.findall(r'\(\*.*?\*\)', st_code, re.DOTALL)  
    comment_lines = sum([comment.count('\n') + 1 for comment in comments])
    comment_density = comment_lines / code_length if code_length > 0 else 0

    print (f"Total lines of code: {code_1ength}")
    print (f"number of input variables: {input_mar_count}")
    print (f"Number of output variables: {outputted var count}")
    print (f"number of loops: {loop_comnt}")
    print (f"conditional judgment quantity: {condition_comunt}")
    print (f"Number of library function calls: {library_function-count}")
    print (f"Cyclomatic Complexity: {cyclomatic_complety}")
    print (f"Code comment density: {comment_density:.2f}")