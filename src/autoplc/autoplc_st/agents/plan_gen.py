from autoplc_st.prompts import plan_gen_prompt
from autoplc_st.agents.clients import (
    autoplc_client_anthropic as client,
    BASE_MODEL as model
)
from dotenv import load_dotenv
import os
from autoplc_st.tools import PromptResultUtil

def plan_gen(current_task_requirement, current_task_code):
    
    code_messages = [{"role": "system", "content":plan_gen_prompt.system_prompt}]


    natural_language_requirement = current_task_requirement
    scl_code = current_task_code
    
    task_prompt = f"""
    自然语言需求：{natural_language_requirement}

    代码：{scl_code}
    你需要根据自然语言需求生成能够为后续的代码生成提供指导的算法流程描述，并根据提供的代码流程进行适当调整。
    """
    
    code_messages.append({"role": "user", "content":plan_gen_prompt.fewshot_prompt(plan_gen_prompt.fewshot_requirement, plan_gen_prompt.fewshot_st_code)})
    code_messages.append({"role": "assistant", "content":plan_gen_prompt.fewshot_feedback(plan_gen_prompt.fewshot_res)})
    code_messages.append({"role": "user", "content":plan_gen_prompt.fewshot_prompt(plan_gen_prompt.fewshot_requirement_2, plan_gen_prompt.fewshot_st_code_2)})
    code_messages.append({"role": "assistant", "content":plan_gen_prompt.fewshot_feedback(plan_gen_prompt.fewshot_res_2)})

    code_messages.append({"role": "user", "content": task_prompt})

    #测试不同模型的生成效果
    model = "deepseek-reasoner"


    response = client.messages.create(
        model=model,  
        max_tokens=16*1024,
        temperature=0.1,
        top_p=0.9,
        messages=code_messages,
    )
    code_messages.append({"role": "assistant", "content":response.choices[0]['message']['content']})
    print(code_messages)
    str1 = PromptResultUtil.message_to_file_str(code_messages)
    print(str1)
    response = response.choices[0]['message']['content']
    return response


def gen_plan_dataset(case_requirement_dir, case_code_dir, case_plan_dir):
    for root, dirs, files in os.walk(case_requirement_dir):
        for file in files:
            requirement_file_path = os.path.join(root, file)
            # 读取第一个目录下文件的内容
            with open(requirement_file_path, 'r', encoding='utf-8') as req_file:
                requirement_content = req_file.read()
            base_name = os.path.splitext(file)[0]
            scl_file_name = base_name + '.st'
            print(scl_file_name)
            # 构建第二个目录中对应的文件路径
            code_file_path = os.path.join(case_code_dir, scl_file_name)
            if os.path.exists(code_file_path):
                # 读取第二个目录下对应文件的内容
                with open(code_file_path, 'r', encoding='utf-8') as code_file:
                    code_content = code_file.read()
            else:
                print(f"在 {case_code_dir} 中未找到对应的文件: {file}")
                code_content = ""
            case_plan = plan_gen(requirement_content, code_content)
            
            plan_file_name = base_name + '.plan'
            plan_file_path = os.path.join(case_plan_dir, plan_file_name)
            # 将需求内容和代码内容写入计划文件

            # print("___________________________\n\n\n")
            # print(case_plan)
            # print("___________________________\n\n\n")

            with open(plan_file_path, 'w', encoding='utf-8') as plan_file:
                plan_file.write(case_plan)
