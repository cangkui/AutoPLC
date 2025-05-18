from dotenv import load_dotenv
load_dotenv()

from common import Config

if __name__ == "__main__":
    # TODO:We need to generate plans at first so that we can use plans as shots in planning agents.
    from autoplc_scl import run_autoplc_scl
    exp_config = Config(config_file="default")
    run_autoplc_scl(benchmark="competition", config=exp_config)
    # run_autoplc_scl(benchmark="lgf", config=exp_config)
