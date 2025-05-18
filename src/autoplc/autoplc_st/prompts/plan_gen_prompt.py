system_prompt = f"""
你是一个SCL代码生成领域的思维链总结专家,擅长根据自然语言需求和示例代码写出比算法流程更粗的思维逻辑。你要设想你在根据提供的需求写出这个算法的实现逻辑，并且根据给出的代码进行调整，但是不要把中间思考的过程写在回答中。
请注意，你要用中文回答
"""
# system_prompt = f"""
# 请分析提供的自然语言需求和代码，生成一个通用的算法思维逻辑流程，该流程应具有普适性，能够指导相似需求的逻辑生成。输出内容仅包含算法的思维逻辑流程，无需其他无关信息。
# 请注意，你要用中文回答
# """


fewshot_requirement = r"""
{
    "title": "Hyperbolic Arctangent Function (ATANH)",
    "description": "ATANH calculates the Arcus Hyperbolic tangent as follows:",
    "type": "FUNCTION",
    "name": "ATANH",
    "input": [
        {{
            "name": "X",
            "type": "REAL",
            "description": "A real number input for which the hyperbolic arctangent is to be calculated."
        }}
    ],
    "output": [],
    "in/out": [],
    "return_value": {{
        "type": "REAL",
        "description": "The hyperbolic arctangent of the input value X.",
        "fields": []
    }}
}
"""

fewshot_requirement_2 = r"""
{
    {
    "title": "Convert Day to Time",
    "description": "DAY_TO_TIME calculates a value (TIME) from the input value\nin days as REAL.",
    "type": "FUNCTION",
    "name": "DAY_TO_TIME",
    "input": [
        {
            "name": "IN",
            "type": "REAL",
            "description": "The input value representing a duration in days that needs to be converted into time."
        }
    ],
    "output": [
        {
            "name": "TIME",
            "type": "TIME",
            "description": "The output value representing the equivalent time format after conversion from the input days."
        }
    ],
    "in/out": [],
    "return_value": {
        "type": "TIME",
        "description": "The TIME value that represents the input duration in days converted to time.",
        "fields": []
    }
}
}
"""



fewshot_st_code = r"""
FUNCTION ATANH : REAL
VAR_INPUT
    X : REAL;
END_VAR
ATANH := LN((1.0 + x)/(1.0 - x)) * 0.5;


(* revision history
hm		12 jan 2007	rev 1.0
	original version

hm		5. jan 2008	rev 1.1
	improved code for better performance

hm	10. mar. 2009		rev 1.2
	real constants updated to new systax using dot

*)
END_FUNCTION

"""


fewshot_st_code_2 = r"""
FUNCTION DAY_TO_TIME : TIME
VAR_INPUT
    IN : REAL;
END_VAR
DAY_TO_TIME := DWORD_TO_TIME(REAL_TO_DWORD(IN * 86400000.0));


(* revision history
hm	4. aug. 2006	rev 1.0
	original release

hm	24. feb. 2009	rev 1.1
	renamed input to IN
*)
END_FUNCTION

"""


fewshot_res = r"""
- 概述
  这个功能块需要实现将二进制数转换为格雷码的功能。格雷码是一种二进制编码，其特点是相邻的两个数之间只有一位二进制数不同，常用于减少数字信号传输中的错误。
- 变量定义
  代码中定义了一类变量：
  - 输入变量：`IN`，表示输入的二进制数，类型为 `BYTE`（8位无符号整数）。

- 主逻辑部分
  -REGION 转换逻辑
     此区域需要实现二进制数到格雷码的转换。格雷码的生成规则是：将二进制数与其右移一位的结果进行按位异或操作（XOR）。具体步骤如下：
     1. 将输入 `IN` 右移一位，得到 `SHR(IN, 1)`。
     2.将 `IN` 与右移后的结果进行按位异或操作，得到格雷码。
     3. 将结果赋值给 `BYTE_TO_GRAY`，作为输出。

  例如，如果 `IN` 的值为 `0b1101`（即十进制的13）：
  - 右移一位后得到 `0b0110`。
  - 进行异或操作：`0b1101 XOR 0b0110 = 0b1011`。
  因此，`BYTE_TO_GRAY` 的输出为 `0b1011`（即十进制的11）。

  这种转换方式确保了相邻的两个二进制数在转换为格雷码后只有一位不同，符合格雷码的特性。
"""

fewshot_res_2 = r"""
    这个需求需要编写一个名为DAY_TO_TIME的函数，用于将输入的天数（REAL 类型）转换为时间（TIME 类型）。转换的逻辑基于一天等于 86,400,000 毫秒的计算。
    变量定义
    输入变量：
    IN：一个 REAL 类型的变量，表示输入的天数。
    无其他变量定义，因为这是一个简单的转换函数。
    主逻辑部分
    -- REGION 转换逻辑：
    函数的核心逻辑是将输入的天数转换为毫秒数，然后再将毫秒数转换为 TIME 类型。具体步骤如下：
    将输入的天数（IN）乘以 86,400,000（一天等于 86,400,000 毫秒），得到一个以毫秒为单位的数值。
    使用 REAL_TO_DWORD 函数将 REAL 类型的毫秒数转换为 DWORD 类型。
    使用 DWORD_TO_TIME 函数将 DWORD 类型的毫秒数转换为 TIME 类型。
    将最终结果赋值给函数返回值 DAY_TO_TIME。
    示例：
    如果输入 IN = 1.0，表示 1 天，计算结果为 86,400,000 毫秒，最终返回 T#1D。
    如果输入 IN = 0.5，表示 0.5 天，计算结果为 43,200,000 毫秒，最终返回 T#12H。
    注意：
    由于 TIME 类型的范围有限（T#0 到 T#24D20H31M23S647MS），如果输入的天数过大，可能会导致溢出。
    输入的天数应为非负数，否则结果可能不符合预期。
"""


def fewshot_prompt(natural_language_requirement, fewshot_st_code):
    return f"""
    自然语言需求：{natural_language_requirement}

    代码：{fewshot_st_code}
    你需要根据自然语言需求生成能够为后续的代码生成提供指导的算法流程描述，并根据提供的代码流程进行适当调整。
    """

def fewshot_feedback(feedback):
    return f"""
    {fewshot_res}
    """

# fewshot_prompt = f"""
#     自然语言需求：{natural_language_requirement}

#     代码：{fewshot_st_code}
#     你需要根据自然语言需求生成能够为后续的代码生成提供指导的算法流程描述，并根据提供的代码流程进行适当调整。
# """
# fewshot_prompt_feedback = f"""

# {feedback}
# """


  
test = f"""
12131
"""
