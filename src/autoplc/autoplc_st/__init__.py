import os
import time
from utils import APIDataLoader
# from autoplc_st.agents import (
#     RetrieverAgent,
#     PlannerAgent,
#     EditorAgent,
#     VerifierAgent,
#     init_team_log_path
# )

def generate_plans():
    from autoplc_st.agents.plan_gen import gen_plan_dataset
    root_path = os.getenv("ROOTPATH")
    case_requirement_dir = os.path.join(root_path, "data/rag_data/st/st_case_requirement")
    # case_requirement_dir = os.path.join(root_path, "src/experiment/datasets/competition/")
    case_code_dir = os.path.join(root_path, "data/rag_data/st/st_case_code")
    case_plan_dir = os.path.join(root_path, "data/rag_data/st/st_case_plan_new")
    gen_plan_dataset(case_requirement_dir, case_code_dir, case_plan_dir)