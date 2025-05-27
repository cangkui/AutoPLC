import os
import pytz
from datetime import datetime
from .retriever_agent import Retriever
from .planner_agent import Modeler
from .editor_agent import LogicComposer
from .verifier_agent import AutoDebugger
from .clients import OpenAIClient,ZhipuAIQAClient
from .api_agent import ApiAgent
from .learner_agent import LearnAgent
from common import ROOTPATH


def init_team_log_path(checkpoint:str = None) -> str:

    if checkpoint:
        # 如果提供了 checkpoint，则使用该路径
        base_folder = checkpoint
        if not os.path.exists(base_folder):
            raise ValueError(f"Checkpoint path {checkpoint} does not exist.")
    else:

        tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(tz)
        date_folder = current_time.strftime("%Y-%m-%d")
        time_folder = current_time.strftime("%H-%M-%S")

        base_folder = ROOTPATH.joinpath(f"output/{date_folder}_{time_folder}")
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

    Retriever.base_logs_folder = base_folder
    Modeler.base_logs_folder = base_folder
    LogicComposer.base_logs_folder = base_folder
    AutoDebugger.base_logs_folder = base_folder
    OpenAIClient.experiment_base_logs_folder = base_folder
    ZhipuAIQAClient.experiment_base_logs_folder = base_folder
        
    return str(base_folder)