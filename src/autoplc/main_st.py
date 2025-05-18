from dotenv import load_dotenv
load_dotenv()

from common import Config

if __name__ == "__main__":
    # TODO:We need to generate plans at first so that we can use plans as shots in planning agents.
    from autoplc_st import run_autoplc_st
    exp_config = Config(config_file="default_st")
    run_autoplc_st(benchmark="oscat", config=exp_config)
    # run_autoplc_scl(benchmark="lgf", config=exp_config)
