import os
import shutil
import time
import json
from autoplc_scl.agents.clients import ClientManager
from autoplc_scl.tools import APIDataLoader, PromptResultUtil
import multiprocessing
from autoplc_scl.agents import (
    Retriever,
    Modeler,
    ApiAgent,
    LogicComposer,
    AutoDebugger,
    LearnAgent,
    init_team_log_path
)
from common import Config, ROOTPATH
import logging
logger = logging.getLogger("autoplc_scl")


root_path = ROOTPATH
def generate_plans():
    from autoplc_scl.agents.plan_gen import gen_plan_dataset
    global root_path
    # root_path = os.getenv("ROOTPATH")
    case_requirement_dir = os.path.join(root_path, "data", "rag_data", "scl", "scl_case_requirement")
    # case_requirement_dir = os.path.join(root_path, "src", "experiment", "datasets", "competition")
    case_code_dir = os.path.join(root_path, "data", "rag_data", "scl", "scl_case_code")
    case_plan_dir = os.path.join(root_path, "data", "rag_data", "scl", "scl_case_plan_experiment")
    gen_plan_dataset(case_requirement_dir, case_code_dir, case_plan_dir)


def decide_is_state_machine_in_lgf():
    from autoplc_scl.agents.plan_gen import figure_state_machine_in_lgf
    global root_path
    # root_path = os.getenv("ROOTPATH")
    dataset_file = os.path.join(root_path, "data", "benchmarks", "lgf_state_machine.jsonl")
    figure_state_machine_in_lgf(dataset_file)

def baseline_in_github_case(model):
    from autoplc_scl.agents.plan_gen import run_baseline_in_github_case
    global root_path
    # root_path = os.getenv("ROOTPATH")
    dataset_file = os.path.join(root_path, "data", "benchmarks", "githubcase.jsonl")
    run_baseline_in_github_case(model, dataset_file)

def run_autoplc_scl(benchmark: str, config: Config):
    global root_path
    root_path = ROOTPATH
    benchmark_file_path = os.path.join(root_path, "data", "benchmarks", f"{benchmark}.jsonl")
    if not os.path.exists(benchmark_file_path):
        logger.error(f"Benchmark file {benchmark_file_path} not found.")
        return

    ClientManager().set_config(config)
    APIDataLoader.init_load(config = config)
    base_folder = init_team_log_path()

    os.makedirs(os.path.join(base_folder, "config"), exist_ok=True)
    shutil.copy(config.config_path, os.path.join(base_folder, "config", "config.yaml"))

    all_agents_start_time = time.time()

    with open(benchmark_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tasks = [json.loads(line) for line in lines]

    for task in tasks:
        autoplc_scl_workflow(task, base_folder, config)

    total_time = time.time() - all_agents_start_time
    logger.info(f"Experiment completed in {total_time:.2f} seconds.")


def autoplc_scl_workflow(
        task: dict, 
        base_folder: str,
        config:Config
    ):

    # 检查是否存在对应的 SCL 文件
    scl_file_path = os.path.join(config.SCL_CODE_DIR, f"{task['name']}.scl")
    if os.path.exists(scl_file_path):
        groundtruth_scl = open(scl_file_path, "r", encoding="utf8").read()
        logger.info(f"Loaded groundtruth scl: {task['name']}")
    else:
        groundtruth_scl = None
        logger.warning(f"Groundtruth scl for {task['name']} not found.")


    # 获取大模型客户端
    openai_client = ClientManager().get_openai_client()
    zhipuai_client = ClientManager().get_zhipuai_client()
    local_api_retriever = ClientManager().get_local_api_retriever()

    os.makedirs(os.path.join(base_folder, f"{task['name']}"), exist_ok=True)

    start_time = time.time()

    # 初始化finally块中的变量
    sample_names = []
    logic_for_this_task = ""
    apis_for_this_task = []

    import traceback
    try:
        ###############    retrieve examples    ################
        retrieved_samples = []
        if not config.RETRIEVE_DISABLED:
            retrieved_samples = Retriever.run_retrieve_case(
                task=task,
                alternatives=config.CASE_ALTERNATIVES,
                zhipuai_client=zhipuai_client
            )
            

        ################ load related algorithms and possible apis ################
        sample_names = [sample["name"] for sample in retrieved_samples]
        logger.info(f"Retrieved samples: {sample_names}")

        related_algorithm = []
        for name in sample_names:
            # algo = PromptResultUtil.get_plan(name=name, code_type="scl")      #为了跑不同版本的plan 需要传plan的版本的名称
            algo = PromptResultUtil.get_plan_diff(
                name=name, 
                plan_version = config.PLAN_SET_NAME
            )
            if algo is not None:
                related_algorithm.append(algo)
            else:
                related_algorithm.append("")

        if sample_names:
            #获取相似案例用到的api
            api_from_similar_cases = APIDataLoader.extract_apis_from_cases(sample_names)    
        else:
            api_from_similar_cases = []

        ################      generate plan      ################
        logic_for_this_task = ""
        if not config.MODELING_DISABLED:
            logic_for_this_task = Modeler.run_modeling_task(
                task=task, 
                retrieved_examples=retrieved_samples,
                related_algorithm=related_algorithm,
                openai_client=openai_client,
                load_few_shots=config.IS_MODELING_FEWSHOT
            )
            # print(f"[INFO] logic for this task:\n {logic_for_this_task}")
            # save logic for this task to file
        
        ################     api recommend      ################
        if not config.APIREC_DISABLED:
            api_recommend, library_func_recommend  = ApiAgent.run_recommend_api(
                task=task, 
                algorithm_for_this_task=logic_for_this_task,
                openai_client=openai_client,
                zhipuai_client=zhipuai_client,
                local_api_retriever=local_api_retriever,
                load_few_shots=config.IS_APIREC_FEWSHOT
            )
        else:
            api_recommend = []
            library_func_recommend = []
            
        apis_for_this_task = list(set(api_recommend + api_from_similar_cases))

        return # 暂时跳过，测试api推荐的效果
        ################      generate SCL      ################
        scl_code = LogicComposer.run_gen_scl(
            task=task,
            retrieved_examples = retrieved_samples,
            related_algorithm = related_algorithm,
            logic_for_this_task= logic_for_this_task,
            apis_for_this_task = apis_for_this_task,
            openai_client = openai_client,
            load_few_shots=config.IS_CODING_FEWSHOT
        )
        first_gen_scl = scl_code

        ################       debug SCL      ################
        if not config.DEBUGGER_DISABLED:
            scl_code = AutoDebugger.run_debugger_with_compiler(
                task=task,
                scl_code=scl_code,
                max_verify_count=config.VERIFY_COUNT,
                openai_client=openai_client,
                load_few_shots=config.IS_DEBUGGER_FEWSHOT
            )
        else:
            # save scl code to file
            code_output_file = os.path.join(base_folder, f"{task['name']}/{task['name']}_{0}.scl")
            logger.info(f"output file is {code_output_file}")
            with open(code_output_file, "w", encoding="utf-8") as fp:
                fp.write(scl_code)
        
        ###############    auto learner      ################
        if not config.AUTOLEARN_DISABLED:

            # 加载history
            if os.path.exists(os.path.join(base_folder, f"{task['name']}", "verify_info.jsonl")):
                try:
                    with open(os.path.join(base_folder, f"{task['name']}", "verify_info.jsonl"), "r", encoding="utf-8") as f:
                        debug_history = [json.loads(line) for line in f.readlines()]
                except Exception as e:
                    logger.error(f"Error decoding JSON from verify_info.jsonl: {e}")
                    debug_history = []
            else:
                debug_history = []

            if groundtruth_scl is not None:
                logger.info("Start auto learner from groundtruth scl.")
                coding_feed_back = LearnAgent.run_learn_from_coding(
                    task=task,
                    prediction_scl=first_gen_scl, # 这里用的是第一次生成的scl代码，因为我们希望模型提高首次生成效率
                    openai_client=openai_client,
                    groundtruth_scl=groundtruth_scl,
                    debug_history=debug_history
                )
                # save feedback json
                with open(os.path.join(base_folder, f"{task['name']}", "coding_feedback.json"), "w", encoding="utf-8") as f:
                    json.dump(coding_feed_back, f, ensure_ascii=False, indent=4)

            

            if len(debug_history) > 0 and groundtruth_scl is not None:
                logger.info("Start auto learner from debug history.")
                debug_feed_back = LearnAgent.run_learn_from_debug(
                    task = task,
                    groundtruth_scl = groundtruth_scl,
                    debug_history = debug_history,
                    openai_client = openai_client
                )
                # save debug feedback json
                with open(os.path.join(base_folder, f"{task['name']}", "debug_feedback.json"), "w", encoding="utf-8") as f:
                    json.dump(debug_feed_back, f, ensure_ascii=False, indent=4)


    except Exception as e:
        logger.exception(f"Error occurred while processing task {task['name']}: {e}")
        logger.exception(e)
    finally :
        # save all intermediate results to file
        try:
            with open(os.path.join(base_folder, f"{task['name']}", "intermediate_results.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "retrieved_samples": sample_names if sample_names else [],
                    "logic_for_this_task": logic_for_this_task if logic_for_this_task else "",
                    "apis_for_this_task": apis_for_this_task if apis_for_this_task else [],
                },fp=f, ensure_ascii=False, indent=4)
        except Exception as e: 
            logger.error(f"Error occurred while saving intermediate results for task {task['name']}: {e}")
            pass
        
    logger.info(f"Task {task['name']} completed in {time.time() - start_time:.2f} seconds.")
