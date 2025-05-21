from dotenv import load_dotenv
load_dotenv()
from common import Config
import logging
from rich.logging import RichHandler

# 初始化 logger
logger = logging.getLogger("autoplc_scl")
logger.setLevel(logging.INFO)

# 仅添加 Rich 控制台输出，不写文件
console_handler = RichHandler(rich_tracebacks=True, markup=True)

logger.handlers.clear()  # 清除旧 handler（避免重复）
logger.addHandler(console_handler)

from autoplc_scl import run_autoplc_scl

if __name__ == "__main__":
    # TODO:We need to generate plans at first so that we can use plans as shots in planning agents.
    
    # read arguments from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="competition", help="benchmark name")
    parser.add_argument("--config", type=str, default="default", help="config name")
    args = parser.parse_args()
    
    run_autoplc_scl(benchmark=args.benchmark, config=Config(config_file=args.config))
