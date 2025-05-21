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

if __name__ == "__main__":
    # TODO:We need to generate plans at first so that we can use plans as shots in planning agents.
    from autoplc_st import run_autoplc_st
    exp_config = Config(config_file="default_st")
    run_autoplc_st(benchmark="oscat", config=exp_config)
    # run_autoplc_scl(benchmark="lgf", config=exp_config)
