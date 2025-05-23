import time
from typing import Tuple, List
import os
from typing import Tuple

from autoplc_st.tools import PromptResultUtil
from autoplc_st.tools import APIDataLoader
from autoplc_st.agents.clients import ClientManager, OpenAIClient
import logging

logger = logging.getLogger("autoplc_st")

class Modeler():
    """
    Modeler类用于生成算法流程描述。它通过运行建模任务来生成指导后续代码生成的算法流程。

    Attributes:
        base_logs_folder (str): 用于存储基础日志文件夹的路径。

    Methods:
        run_modeling_task(task: dict, retrieved_examples: List[dict], related_algorithm: List[dict]) -> Tuple[str, int, int]:
            运行建模任务以生成算法流程描述。该方法使用任务信息、检索到的示例和相关算法来生成算法流程描述。
    """

    base_logs_folder = None

    @classmethod
    def run_modeling_task(cls, 
            task: dict,
            retrieved_examples: List[dict],
            related_algorithm: List[dict],
            openai_client : OpenAIClient,
            load_few_shots: bool = True,
            ) -> str:
        """
        运行建模任务以生成算法流程描述。

        参数:
        - task: 包含任务信息的字典。
        - retrieved_examples: 检索到的示例列表，每个示例是一个字典。
        - related_algorithm: 相关算法的列表，每个算法是一个字典。
        - openai_client : LLM客户端
        - load_few_shots: 是否加载few-shot示例。
        
        返回:
        - algorithm_for_this_task: 生成的算法流程描述字符串。
        """

        logger.info("开始执行planner")
        # 记录开始时间，用于计算执行时长
        start_time = time.time()
        logger.info("\n------------------------------------------------\n")

        # 构造消息列表，包括系统提示和用户要求
        messages = [ 
            # {"role": "system", "content": state_machine_prompt}, 
            # {"role": "system", "content": system_prompt_state_machine}
            {"role": "system", "content": state_machine_prompt_en}
            # {"role": "user", "content": str(task)}
        ]

        # 加载few-shot示例
        if load_few_shots:
            fewshots = cls.load_plan_gen_shots(retrieved_examples, related_algorithm)
            messages.extend(fewshots)

        messages.append({"role": "user", "content": plan_shots_prompt_en.format(task_requirements = str(task)) })
        # print(messages)

        # 调用模型生成规划
        response = openai_client.call(
            messages,
            task_name=task["name"],
            role_name="planner"
        )

        # 打印结束生成规划的消息和执行时间
        # print("\n------------------------------------------------\n")
        # print("planner已执行完毕")

        algo = cls.extract_content(response)

        # 返回规划内容和令牌使用数量
        logger.info(f"End - generate plan - Execution Time: {(time.time() - start_time):.6f} seconds")

        return algo

    @classmethod
    def extract_content(cls,response) -> str:
        """
        该函数用于从响应中提取生成的内容。

        参数:
        - response: AI模型的响应对象。

        返回:
        - algo: 提取出的算法流程描述字符串。
        """
        if hasattr(response, 'choices'):
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                algo = choice.message.content
            else:
                algo = choice['message']['content']
        else:
            algo = response.content[0].text

        return algo

    @classmethod
    def load_plan_gen_shots(cls, examples: List[dict], algorithms: List[str]):
        """
        Load few-shot for plan
        """
        formatted = []
        shot_prompt = plan_shots_prompt_en
        for example, algo in zip(examples, algorithms):
            icl_input = shot_prompt.format(
                task_requirements=str(example['json_desc'])
                # algorithm_process = "",
                # algorithm_process=f"<! -- A plan that you can refer to when coding -->\n<Plan>\n{algo}\n</Plan>\n"
            )
            icl_output = PromptResultUtil.remove_braces(algo)
            formatted.append({"role": "user", "content": icl_input})
            formatted.append({"role": "assistant", "content": icl_output})
        # print("---------------------------------------------------------------------------------------------------------------------")
        # print(formatted)
        return formatted



plan_shots_prompt_en = """Here is the input, structured in XML format:
<! -- st programming task requirements to be completed -->
<Task_Requirements>
{task_requirements}
</Task_Requirements>
"""

state_machine_prompt_en = """
You are a professional PLC engineer working in a Codesys-based development environment.

Please first determine whether the given requirement is a **process control task** or a **data processing task**.

For **process control tasks**:
    1. Analyze the possible control states involved in the process.
    2. Identify clear and deterministic state transition events or conditions.
    3. Describe the algorithmic workflow and control logic in a structured, human-readable manner. Do not output pseudocode or any form of code!

For **data processing tasks**:
    1. Describe the algorithmic workflow and computational steps. Do not output pseudocode or any form of code!

Notes:
- Maintain a professional and systematic tone, in line with IEC 61131-3 Structured Text conventions.
- Consider initialization and persistent variable behavior carefully to avoid data loss or unexpected resets.
- In the absence of any triggering event, the system should remain in the current state and continue executing its associated logic.
- Only include exception handling logic if it is explicitly mentioned in the requirement.
- Be mindful of typical Codesys task cycle execution and ensure that the logic is consistent with real-time execution constraints.
""".strip()

state_machine_prompt = """
你是PLC工程师。

请你先判断该需求是顺序控制任务还是数据处理任务。

对顺序控制任务:
    1.分析有哪些状态。 
    2.分析状态转移事件。
    3.给出算法流程,不要输出伪代码或者其他任何形式的代码内容!

对数据处理任务:
    1.给出算法流程,不要输出伪代码或者其他任何形式的代码内容!

注意事项:
- 保持语言专业严谨，符合西门子 S7-1200/1500 系列 PLCs 的规范。
- 谨慎进行初始化操作,防止误删重要数据。
- 如果没有发生转移事件，应当维持当前状态的动作。 
- 如果需求中明确要求处理异常情况,则添加错误处理逻辑,否则无需考虑。
""".strip()


new_system_prompt = """
用中文描述你解决这个问题的算法步骤，不输出任何伪代码。告诉我解决这个问题需要特别注意的方面。
Important:
- 保持语言专业严谨，符合PLC编程语言的规范。
- 不允许对In_Out类型参数进行初始化操作。
- 无需考虑数据类型转换。
- Static类型变量的初始化操作需要谨慎,防止误删其中保存的重要数据(如数据库的当前存储状态)
- 如果需要检测“跳变”、“上升沿”或“下降沿”，你应当使用【静态变量】保存每个周期的输入状态state，再与下一周期的输入进行比较(`IF newState and not oldsState`)以确定是否发生跳变。
- 类似灌装生产线的控制系统通常被称为实时控制系统。因此在编程时，注意每个扫描周期检查和保持历史状态，确保在没有新信号时维持现有状态。意味着在每个扫描周期内，PLC会记住上一周期的状态，并在没有新输入信号的情况下维持这个状态。这样可以确保系统在没有新的输入信号时不会改变当前的输出状态。以灌装生产线为例，当传感器检测到瓶子到达灌装位置时，启动灌装阀门。只有在操作员确认灌装完成后，才关闭灌装阀门。如果没有操作员的确认信号，阀门应该保持打开状态。
""".strip()

system_prompt_state_machine = f"""
你是一个st代码生成领域的思维链总结专家,擅长根据自然语言需求和示例代码写出比算法流程更粗的思维逻辑。你要设想你在根据提供的需求写出这个算法的实现逻辑。不要定义非必要的变量。
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



system_prompt_copy_relate_plan = f"""
你是一个st代码生成领域的思维链总结专家,擅长根据自然语言需求和示例代码写出比算法流程更粗的思维逻辑。
你要设想你在根据提供的需求写出这个算法的实现逻辑。不要定义非必要的变量。
# 请注意，你要用中文回答
"""


old_state_machine_and_prompt = """
先判断是过程控制任务还是数据处理任务。
对过程控制任务:
    1.分析有哪些状态。 
    2.分析状态转移事件。
    3.给出算法流程,不要输出代码
对数据处理任务:
    1.给出算法流程,不要输出代码
IMPORTANT:
- 保持语言专业严谨，符合PLC编程语言的规范。
- 考虑数据类型转换,必要时将Byte、Int转换为Real再计算。
- 如果没有发生转移事件，维持当前状态的动作。 
- 保持语言专业严谨，符合PLC编程语言的规范。
- 不允许对In_Out类型参数进行初始化操作。
- Static类型变量的初始化操作需要谨慎,防止误删其中保存的重要数据(如数据库的当前存储状态)。
- 如果需要检测“跳变”、“上升沿”或“下降沿”，你应当使用【静态变量】保存每个周期的输入状态state，再与下一周期的输入进行比较(`IF newState and not oldsState`)以确定是否发生跳变。
""".strip()



list_problem_prompt = """
Role: 你是一名专家st（Structured Control Language）算法细化分析者，负责找出算法流程中需要进一步细化描述的部分，并进行细化，最终得到新的算法流程。不要输出伪代码，用自然语言描述！
Goals:
1. 根据提供的需求，找出需要进一步的细化的算法流程片段。
2. 细化需要进一步细化的算法流程片段。
3. 得到细化后的算法流程。
4. 确保算法描述尽可能详细、准确，涵盖尽可能多的信息，并易于理解。
Workflow:
1. 仔细阅读并理解提供的需求。
2. 仔细阅读并理解提供的算法流程。
3. 寻找流程中可以被进一步细化的部分，如数组排序等。
4. 详细地列出算法流程中可以被细化的部分。
5. 对列出的可细化部分进行细化。
6. 输出经过细化后的算法流程
Constraints:
1. 回答不要出现伪代码或者其他任何形式的代码内容，必须以自然语言描述算法流程。
Important:
1. 对于提出有待细化的部分，给出具体的细化措施！
2. 展开具体讲讲细化的具体步骤是什么。
3. 确保你的细化措施仍然遵循算法流程。
4. 确保你的细化措施满足任务需求。
5. 细化所采用的算法要尽可能从编码方面简单，不用过于考虑时间复杂度，例如对于排序，优先采用插入排序算法而非快速排序算法。
6. 利用原流程的信息以及细化的流程片段信息，将整合的算法流程写得尽可能细节，涵盖尽可能多的信息，包含你所知的所有信息！！！！
7. 整合的算法流程要包含所有原算法流程中出现的变量！！
8. 不要输出伪代码或者其他任何形式的代码内容，用自然语言描述算法流程！

Output Format:
```plaintext
----有待细化的算法流程片段以及细化操作流程----
1. [需要细化的部分]：
- [细化内容1]：
- [细化内容2]：
- [细化内容n]：
...
K. [需要细化的部分]：
- [细化内容1]：
- [细化内容2]：
- [细化内容n]：
--------
----细化后的算法流程----
// 根据需求撰写详细的自然语言算法描述，覆盖包括但不限于以下部分：
- [输入和输出变量声明]
- [临时变量声明]
- [常量声明]
- [初始化步骤]
- [输入输出验证]
- [主逻辑1]
- [主逻辑2]
...
- [主逻辑n]
- [结果输出]
--------
```
""".strip()
