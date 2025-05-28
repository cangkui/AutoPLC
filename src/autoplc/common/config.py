import os
import yaml
from typing import List, Dict, Any
from pathlib import Path


ROOTPATH = Path(__file__).resolve().parents[3]


class Config:
    def __init__(self, config_file: str = "default"):
        print("config path is", config_file)
        self.ROOTPATH = ROOTPATH
        config_path = self.ROOTPATH.joinpath(f"src/config/{config_file}.yaml")
        self._config = self._load_config(config_path)
        self.config_path = config_path
        self._resolve_environment_variables()
            
    @classmethod
    def load_from_absolute_path(cls, abs_path: str) -> "Config":
        """
        从给定的绝对路径加载配置，返回一个新的 Config 实例。
        """
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Configuration file not found: {abs_path}")
        
        instance = cls.__new__(cls)  # 不调用 __init__
        instance.ROOTPATH = ROOTPATH
        instance._config = instance._load_config(abs_path)
        instance.config_path = abs_path
        instance._resolve_environment_variables()
        return instance

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _resolve_environment_variables(self):
        """解析配置中的环境变量引用"""
        # root_path = os.getenv("ROOTPATH")
        # if not root_path:
        #     raise ValueError("ROOTPATH environment variable is not set")
        
        # 递归解析所有字符串值中的环境变量
        def resolve(obj):
            if isinstance(obj, str):
                return obj.format(
                    ROOTPATH=self.ROOTPATH
                )
            elif isinstance(obj, dict):
                return {k: resolve(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve(item) for item in obj]
            return obj
        
        self._config = resolve(self._config)
    
    @property
    def INSTRUCTION_DIR(self) -> Path:
        return Path(self._config["environment"]["instruction_dir"])

    @property
    def SHOT_DATA_DIR(self) -> Path:
        return Path(self._config["environment"]["shot_data_dir"])
    
    @property
    def LOG_DIR(self) -> Path:
        return Path(self._config["environment"]["log_dir"])
    
    @property
    def SCL_PLAN(self) -> Path:
        return Path(self._config["environment"]["scl_plan"])
    
    @property
    def SCL_CODE_DIR(self) -> str:
        return self._config["environment"]["scl_code_dir"]
    
    @property
    def ST_PLAN(self) -> Path:
        return Path(self._config["environment"]["st_plan"])
    
    @property
    def ST_CODE_DIR(self) -> str:
        return self._config["environment"]["st_code_dir"]

    @property
    def CODESYS_DIR(self) -> str:
        return self._config["environment"]["coedsys_dir"]
    
    @property
    def model(self) -> str:
        return self._config["model"]["name"]
    
    @property
    def max_tokens(self) -> int:
        return self._config["model"]["max_tokens"]
    
    @property
    def temperature(self) -> float:
        return self._config["model"]["temperature"]
    
    @property
    def top_p(self) -> float:
        return self._config["model"]["top_p"]
    
    @property
    def retrieve_model(self) -> str:
        return self._config["retrieve"]["model"]
    
    @property
    def retrieve_temperature(self) -> float:
        return self._config["retrieve"]["temperature"]
    
    @property
    def retrieve_top_p(self) -> float:
        return self._config["retrieve"]["top_p"]
    
    @property
    def PLAN_SET_NAME(self) -> str:
        return self._config["modeler"]["plan_set_name"]
    
    @property
    def VERIFY_COUNT(self) -> int:
        return self._config["verify"]["count"]
    
    @property
    def CODESYS_IP(self) -> List[str]:
        return self._config["verify"]["codesys_ip"]
    
    @property
    def openness_binding_base_url(self) -> int:
        return self._config["verify"]["openness"] \
            if self._config["verify"]["openness"] else "http://localhost:9000"
    
    @property
    def CASE_ALTERNATIVES(self) -> List[str]:
        return self._config["retrieval"]["case_alternatives"]
    
    @property
    def INSTRUCTION_PATH(self) -> List[Path]:
        return [Path(path) for path in self._config["instruction"]["path"]]
    
    @property
    def FUNCTION_USAGE_PATH(self) -> Path:
        return Path(self._config["instruction"]["function_usage_path"])

    @property
    def INSTRUCTION_SCORE_THRESHOLD(self) -> float:
        return self._config["instruction"]["score_threshold"]
    
    @property
    def INSTRUCTION_TOP_K(self) -> int:
        return self._config["instruction"]["top_k"]
    
    @property
    def BM25_MODEL(self) -> str:
        return self._config["retrieval"]["b25_model"]
    
    @property
    def RETRIEVE_DISABLED(self) -> bool:
        return self._config["workflow"]["retrieve_disabled"]
    
    @property
    def MODELING_DISABLED(self) -> bool:
        return self._config["workflow"]["modeling_disabled"]
    
    @property
    def DEBUGGER_DISABLED(self) -> bool:
        return self._config["workflow"]["debugger_disabled"]

    @property
    def APIREC_DISABLED(self) -> bool:
        return self._config["workflow"]["api_rec_disabled"]
    
    @property
    def AUTOLEARN_DISABLED(self) -> bool:
        return self._config["workflow"]["auto_learn_disabled"]

    @property
    def IS_CODING_FEWSHOT(self) -> bool:
        return self._config["workflow"]["is_coding_fewshot"]
    
    @property
    def IS_RETRIEVE_FEWSHOT(self) -> bool:
        return self._config["workflow"]["is_retrieve_fewshot"]
    
    @property
    def IS_MODELING_FEWSHOT(self) -> bool:
        return self._config["workflow"]["is_modeling_fewshot"]
    
    @property
    def IS_DEBUGGER_FEWSHOT(self) -> bool:
        return self._config["workflow"]["is_debugger_fewshot"]
    
    @property
    def IS_APIREC_FEWSHOT(self) -> bool:
        return self._config["workflow"]["is_api_rec_fewshot"]
    
    @property
    def IS_AUTOLEARN_FEWSHOT(self) -> bool:
        return self._config["workflow"]["is_auto_learn_fewshot"]

    def __repr__(self) -> str:
        return yaml.dump(self._config, sort_keys=False)