import os
import time
import json
from autoplc_scl.agents.clients import ClientManager
from autoplc_scl.tools import APIDataLoader, PromptResultUtil
import multiprocessing
from autoplc_scl.agents import (
    Retriever,
    Modeler,
    LogicComposer,
    AutoDebugger,
    init_team_log_path
)
from common import Config, ROOTPATH


root_path = ROOTPATH

def generate_plans():
    from autoplc_scl.agents.plan_gen import gen_plan_dataset
    global root_path
    # root_path = os.getenv("ROOTPATH")
    case_requirement_dir = os.path.join(root_path, "data/rag_data/scl/scl_case_requirement")
    # case_requirement_dir = os.path.join(root_path, "src/experiment/datasets/competition/")
    case_code_dir = os.path.join(root_path, "data/rag_data/scl/scl_case_code")
    case_plan_dir = os.path.join(root_path, "data/rag_data/scl/scl_case_plan_experiment")
    gen_plan_dataset(case_requirement_dir, case_code_dir, case_plan_dir)


def decide_is_state_machine_in_lgf():
    from autoplc_scl.agents.plan_gen import figure_state_machine_in_lgf
    global root_path
    # root_path = os.getenv("ROOTPATH")
    dataset_file = os.path.join(root_path, "data/benchmarks/lgf_state_machine.jsonl")
    figure_state_machine_in_lgf(dataset_file)

def baseline_in_github_case(model):
    from autoplc_scl.agents.plan_gen import run_baseline_in_github_case
    global root_path
    # root_path = os.getenv("ROOTPATH")
    dataset_file = os.path.join(root_path, "data/benchmarks/githubcase.jsonl")
    run_baseline_in_github_case(model, dataset_file)

def run_autoplc_scl(
        benchmark: str,
        config:Config
    ):
    global root_path
    root_path = ROOTPATH

    benchmark_file_path = root_path.joinpath(f"data/benchmarks/{benchmark}.jsonl")
    if not os.path.exists(benchmark_file_path):
        print(f"Benchmark file {benchmark_file_path} not found.")
        return

    # init
    ClientManager().set_config(config)
    APIDataLoader.init_load(code_type="scl")
    base_folder = init_team_log_path()

    # workflow
    all_agents_start_time = time.time()

    with open(benchmark_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tasks = [json.loads(line) for line in lines]
        # tasks = [json.loads(line) for line in lines]

    for task in tasks:
        autoplc_scl_workflow(
            task, 
            base_folder,
            config
    )

    total_time = time.time() - all_agents_start_time
    print(f"Experiment completed in {total_time:.2f} seconds.")


def autoplc_scl_workflow(
        task: dict, 
        base_folder: str,
        config:Config
    ):

    retrieve_disabled = config.RETRIEVE_DISABLED
    modeling_disabled = config.MODELING_DISABLED
    debugger_disabled = config.DEBUGGER_DISABLED
    plan_set_name = config.PLAN_SET_NAME

    # 获取大模型客户端
    openai_client = ClientManager().get_openai_client()
    zhipuai_client = ClientManager().get_zhipuai_client()

    os.makedirs(os.path.join(base_folder, f"{task['name']}"), exist_ok=True)

    start_time = time.time()

    import traceback
    try:
        ###############    retrieve examples    ################
        retrieved_samples = []
        if not retrieve_disabled:
            retrieved_samples = Retriever.run_retrieve_case(task=task,alternatives=config.CASE_ALTERNATIVES,zhipuai_client=zhipuai_client)

        ################ load related algorithms ################
        sample_names = [sample["name"] for sample in retrieved_samples]

        related_algorithm = []
        for name in sample_names:
            # algo = PromptResultUtil.get_plan(name=name, code_type="scl")      #为了跑不同版本的plan 需要传plan的版本的名称
            algo = PromptResultUtil.get_plan_diff(name=name, plan_version = plan_set_name)
            if algo is not None:
                related_algorithm.append(algo)
            else:
                related_algorithm.append("")

        ################      generate plan      ################
        algorithm_for_this_task = ""
        if not modeling_disabled:
            algorithm_for_this_task = Modeler.run_modeling_task(task=task, 
            retrieved_examples=retrieved_samples,
            related_algorithm=related_algorithm,
            openai_client=openai_client
            )

        ################      generate code      ################
        scl_code = LogicComposer.run_gen_scl(
            task=task,
            retrieved_examples=retrieved_samples,
            related_algorithm=related_algorithm,
            algorithm_for_this_task=algorithm_for_this_task,
            openai_client=openai_client
        )

        ################       verify code      ################
        if not debugger_disabled:
            scl_code = AutoDebugger.run_debugger_with_compiler(
                task=task,
                scl_code=scl_code,
                max_verify_count=config.VERIFY_COUNT,
                openai_client=openai_client
            )
    
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
    
    print(f"Task {task['name']} completed in {time.time() - start_time:.2f} seconds.")
