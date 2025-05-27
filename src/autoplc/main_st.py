from dotenv import load_dotenv
load_dotenv()

from common import Config
import logging
from rich.logging import RichHandler
# 初始化 logger
logger = logging.getLogger("autoplc_st")
logger.setLevel(logging.INFO)

# 仅添加 Rich 控制台输出，不写文件
console_handler = RichHandler(rich_tracebacks=True, markup=True)

logger.handlers.clear()  # 清除旧 handler（避免重复）
logger.addHandler(console_handler)

from autoplc_st import run_autoplc_st
import argparse
if __name__ == "__main__":
    # TODO:We need to generate plans at first so that we can use plans as shots in planning agents.
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="oscat", help="benchmark name")
    parser.add_argument("--config", type=str, default="default_st", help="config name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="checkpoint where the task is saved")
    args = parser.parse_args()
    
    run_autoplc_st(benchmark=args.benchmark, config=Config(config_file=args.config), checkpoint_dir=args.checkpoint_dir, max_workers=1)
