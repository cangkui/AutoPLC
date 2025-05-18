import json
import pytz
from datetime import datetime
from pathlib import Path
import os
from autoplc_scl.tools import PromptResultUtil
from autoplc_scl.agents.clients import ClientManager
from common import ROOTPATH

def fill_fewshot(natural_language_requirement, scl_code):
    return f"""
    自然语言需求：{natural_language_requirement}

    代码：{scl_code}
    你需要根据自然语言需求生成能够为后续的代码生成提供指导的算法流程描述，并根据提供的代码流程进行适当调整。
    """    

def plan_gen(current_task_requirement, current_task_code) -> str:
    """
    该函数生成一个算法流程描述，用于指导后续的代码生成。
     
    参数:
    - current_task_requirement: 当前任务的自然语言需求描述。
    - current_task_code: 当前任务的SCL代码。
    
    返回:
    - response: 生成的算法流程描述。
    """
    
    code_messages = [{"role": "system", "content":system_prompt_state_machine}]
    
    # fewshot_context1 = fill_fewshot(fewshot_prompt_req_1, fewshot_prompt_code_1)
    # code_messages.append({"role": "user", "content": fewshot_context1})
    # code_messages.append({"role": "assistant", "content": fewshot_prompt_plan_1})

    natural_language_requirement = current_task_requirement
    scl_code = current_task_code

    task_prompt = f"""
    自然语言需求：{natural_language_requirement}

    代码：{scl_code}
    你需要根据自然语言需求生成能够为后续的代码生成提供指导的算法流程描述，并根据提供的代码流程进行适当调整。
    """

    code_messages.append({"role": "user", "content": task_prompt})

    #测试不同模型的生成效果
    model = "deepseek-chat"
    openai_client = ClientManager().get_openai_client()
    response = openai_client.messages.create(
        model=model,  
        max_tokens=16*1024,
        temperature=0.1,
        top_p=0.9,
        messages=code_messages,
    )
    code_messages.append({"role": "assistant", "content":response.choices[0]['message']['content']})
    # print(code_messages)
    str1 = PromptResultUtil.message_to_file_str(code_messages)
    print(str1)

    response = response.choices[0]['message']['content']
    return response

def is_st_machine(requirement_content) -> str:
    """
    判断给定的需求内容是否为状态机。

    该函数接收一个需求内容字符串，并通过调用语言模型来判断该需求是否可以被视为状态机。
    状态机是一种用于描述系统行为的模型，通常由一组状态和状态之间的转换规则组成。

    参数:
    - requirement_content: 包含需求内容的字符串。

    返回:
    - response: 一个字符串，表示模型对需求内容是否为状态机的判断结果。
    """

    code_messages = [{"role": "system", "content":is_state_machine}]
    code_messages.append({"role": "user", "content": requirement_content})
    model = "gpt-4o-mini"
    openai_client = ClientManager().get_openai_client()
    response = openai_client.messages.create(
        model=model,  
        max_tokens=16*1024,
        temperature=0.1,
        top_p=0.9,
        messages=code_messages,
    )    # print(code_messages)

    response = response.choices[0]['message']['content']
    return response


def baseline_github(input_model, requirement_content) -> str:
    """
    baseline_github 函数用于根据给定的模型和需求内容生成响应。

    该函数接收一个模型名称和需求内容字符串，构造消息列表并调用语言模型生成响应。
    生成的响应将作为字符串返回。

    参数:
    - input_model: 用于生成响应的模型名称。
    - requirement_content: 包含需求内容的字符串。

    返回:
    - response: 生成的响应字符串。
    """
    code_messages = [{"role": "system", "content":baseline_githubCase}]
    code_messages.append({"role": "user", "content": requirement_content})
    model = input_model

    openai_client = ClientManager().get_openai_client()
    response = openai_client.messages.create(
        model=model,  
        max_tokens=16*1024,
        temperature=0.1,
        top_p=0.9,
        messages=code_messages,
    )    # print(code_messages)

    response = response.choices[0]['message']['content']
    return response


def gen_plan_dataset(case_requirement_dir, case_code_dir, case_plan_dir):
    """
    根据用例需求和代码生成计划数据集。

    遍历需求目录下的所有文件，读取每个需求文件的内容，并尝试读取对应代码文件的内容（如果存在），
    然后将两者作为输入生成一个计划，并将该计划写入计划目录下的相应文件中。

    参数:
    case_requirement_dir (str): 用例需求文件所在的目录路径。
    case_code_dir (str): 用例代码文件所在的目录路径。
    case_plan_dir (str): 生成的计划文件将要存放的目录路径。
    """
    
    #创建一个关于时间戳的文件夹
    tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(tz)
    date_folder = current_time.strftime("%Y-%m-%d")
    time_folder = current_time.strftime("%H-%M-%S")

    base_folder = os.path.join(
        case_plan_dir, f"{date_folder}_{time_folder}"
    )
    print(base_folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    # 遍历需求目录及其子目录
    for root, dirs, files in os.walk(case_requirement_dir):
        files.sort()
        for file in files:
            # 构建需求文件的完整路径
            requirement_file_path = os.path.join(root, file)

            # 读取第一个目录下文件的内容
            with open(requirement_file_path, 'r', encoding='utf-8') as req_file:
                requirement_content = req_file.read()
            # 获取文件名（不带后缀）
            base_name = os.path.splitext(file)[0]
            # 构建对应的.scl文件名
            scl_file_name = base_name + '.scl'
            print(scl_file_name)

            # 构建第二个目录中对应的文件路径
            code_file_path = os.path.join(case_code_dir, scl_file_name)
            # 检查对应的代码文件是否存在
            if os.path.exists(code_file_path):
                # 读取第二个目录下对应文件的内容
                with open(code_file_path, 'r', encoding='utf-8') as code_file:
                    code_content = code_file.read()
            else:
                # 如果未找到对应的代码文件，打印提示信息并设置code_content为空字符串
                print(f"在 {case_code_dir} 中未找到对应的文件: {file}")
                code_content = ""

            # 根据需求内容和代码内容生成计划
            case_plan = plan_gen(requirement_content, code_content)
            
            # 将生成的计划内容写入计划文件
            # 构建计划文件名和路径
            plan_file_name = base_name + '.plan'
            plan_file_path = os.path.join(base_folder, plan_file_name)
            with open(plan_file_path, 'w', encoding='utf-8') as plan_file:
                plan_file.write(case_plan)

def figure_state_machine_in_lgf(dataset_file):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line.strip())
            st_res = is_st_machine(line)
            print(json_data['name'], st_res + " ")

def run_baseline_in_github_case(model, dataset_file):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line.strip())
            st_res = baseline_github(model, line)
            print(json_data['name'], st_res + " ")



# system_prompt = f"""
# 请分析提供的自然语言需求和代码，生成一个针对于当前任务的算法思维逻辑流程。输出内容仅包含算法的思维逻辑流程，无需其他无关信息。流程可采用步骤序号的形式呈现，清晰地描述从需求分析到代码实现的关键思考步骤。
# 请注意，你要用中文回答
# """

# system_prompt = f"""
# 你是一个SCL代码生成领域的思维链总结专家,擅长根据自然语言需求和示例代码写出比算法流程更粗的思维逻辑。你要设想你在根据提供的需求写出这个算法的实现逻辑，并且根据给出的代码进行调整。不要定义非必要的变量。
# 请注意，你要用中文回答
# """

# system_prompt = f"""
# 你是一个SCL电气工程师，给定一段需求以及你针对这个需求写出的SCL代码，你要提供你在编写这段代码来完成给定的需求时的思维逻辑。
# 注意，你要用中文回答
# """

system_prompt_state_machine = f"""
你是一个SCL代码生成领域的思维链总结专家,擅长根据自然语言需求和示例代码写出比算法流程更粗的思维逻辑。你要设想你在根据提供的需求写出这个算法的实现逻辑。不要定义非必要的变量。
首先你要判断需求是过程控制任务还是数据处理任务。过程控制任务有状态转换逻辑，而数据处理任务通常是通用的功能函数。这两种任务有不同的分析流程。分析流程如下：
对于过程控制任务:
    1.首先你需要分析，整个过程控制会涉及到哪些状态。 
    2.在这些状态的基础上，你需要分析状态之间的状态转移事件。
    3.在以上两点的基础上，给出算法流程。
对数据处理任务:
    1.给出算法流程。
注意事项:
    1.你需要保持语言专业严谨性，要符合PLC编程语言的规范。
    2.对于过程控制任务，如果没有发生转移事件，维持当前状态的动作。 
    3.如果题目没有明确要求进行错误处理,则无需考虑错误处理逻辑,只有在有村无码出现的时候，再根据具体的错误情况返回错误状态码。

特别注意：
    你给出的是算法流程，不要输出伪代码或者其他任何形式的代码内容!
"""


system_prompt = f"""
你是一个SCL电气工程师，给定一段需求，你要提供你在编写这段代码来完成给定的需求时的思维逻辑。为了让你的思维逻辑尽可能正确，还提供了这个需求对应的SCL代码，你可以参考代码解决问题的逻辑思路，但是不要太过于具体，你要从你提供的思维逻辑能够对类似的需求起到指导作用的角度来写这个需求的解决问题的思维逻辑，但是也不用想太多，从提供的信息出发，一定要避免过度设计，尽量不要写伪代码。
注意，你要用中文回答
"""

# ROOTPATH = os.getenv("ROOTPATH")

req_context = ""
code_context = ""
fs_req = ROOTPATH.joinpath("/data/rag_data/scl/scl_case_requirement")
fs_code = ROOTPATH.joinpath("/data/rag_data/scl/scl_case_code")


with open(os.path.join(fs_req, "FB_ColorLightControl.json"), 'r', encoding='utf8') as f:
    req_context = f.read()
with open(os.path.join(fs_code, "FB_ColorLightControl.scl"), 'r', encoding='utf8') as f:
    code_context = f.read()

fewshot_prompt_req_1 = f"""
{req_context}
"""

fewshot_prompt_code_1 = f"""
{code_context}
"""

fewshot_prompt_plan_1 = f"""
这个需求是要实现一个名为“FB_ColorLightControl”的功能块，用于控制一个彩灯，可能是一个交通信号灯模拟。它有一个控制按钮输入，三个输出分别控制绿灯、红灯和黄灯的亮灭。
变量定义
从给定的函数参数可以看出，需要定义以下几类变量：
- 输入变量：controlButton（控制按钮）
- 输出变量：greenLight（绿灯）、redLight（红灯）、yellowLight（黄灯）
除了给定的变量之外，还需要自己定义一些变量来标记中间值。暂定需要这些变量：
- 内部变量：lightState（灯的状态，用于记录当前是哪种灯亮）、lastControlButtonState（上一次控制按钮的状态，用于检测边缘）
在进行变量定义之后，我设计代码的主要实现逻辑：
主逻辑部分 
-- REGION Validation OF INPUT and OUTPUT
    此区域包含对输入和输出进行验证的逻辑，以确保它们在使用前是有效的。
-- REGION main logic   
    此区域包含控制彩灯状态的主要逻辑。首先，它检查控制按钮是否被按下（即输入信号为真）并且上一次的状态为未按下（即边缘检测）。如果条件为真，则增加灯的状态变量。如果灯的状态变量大于5，则重置为1，因为只有5种可能的灯的状态。然后，根据灯的状态变量，使用CASE语句来设置绿灯、红灯和黄灯的输出。每种状态都对应一种特定的灯的组合。
-- REGION Writing TO outputs   
    此区域包含将内部变量的值写入输出变量的逻辑。不过我认为，这个区域可能不需要额外的逻辑，因为在CASE语句中已经直接设置了输出。
"""

fewshot_prompt_req_2 = f"""
{req_context}
"""

fewshot_prompt_code_2 = f"""
{code_context}
"""

fewshot_prompt_plan_2 = f"""

"""


fewshot_prompt_feedback = f"""

"""

is_state_machine = f"""
我会给你一个这样格式的需求，你给我判断其是否具有状态机的特性。对于有跟状态信息强烈相关的表述或者信息的时候，你要认定其为状态机任务，输出true。
你一定要用true或者false表示你的判断，不要输出其他的信息，true表示你认为是，反之不是。
"""

baseline_githubCase = f"""
你是一个scl代码编写专家，会根据json格式的需求来编写对应的scl代码。你会收到json形式的需求，然后只生成scl代码，不会生成其他信息。
"""