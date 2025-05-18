import os
import sys
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # TODO:We need to generate plans at first so that we can use plans as shots in planning agents.
    #旧模型：DeepSeek-V2.5、Qwen2.5-Coder-Instruct（7B）、GPT-4o、GLM-4-Plus、Claude-3.5-Sonnet和 Llama-3.1-Instruct（8B）
    #新模型（智增增）：deepseek-coder qwen2.5-7b-instruct gpt-4o glm-4-plus	claude-3-5-sonnet-20241022 llama3.1-8b-instruct
    from autoplc_scl import generate_plans
    # run_autoplc_scl(benchmark="competition", plan_disabled=True)
    # run_autoplc_scl(benchmark="lgf", plan_disabled=True)
    generate_plans()
    # run_autoplc_scl(benchmark="lgf", plan_disabled=False)
    # run_autoplc_scl(benchmark="scl_total", plan_disabled=False, plan_set_name="scl_case_plan_NOfewshot")
    # run_autoplc_scl(benchmark="scl_total", plan_disabled=False, plan_set_name="scl_case_plan_oneshot")
    #run_autoplc_scl(benchmark="competition", plan_disabled=False, plan_set_name="scl_case_plan_0312")
    