import json
import os
import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi
import numpy as np
# from zhipuai import ZhipuAI
from anthropic import Anthropic
from openai import OpenAI
from typing import List, Union
from common import Config
from autoplc_st.tools import PromptResultUtil
from anthropic.types.message import Message
# from zhipuai.types.chat.chat_completion import Completion
from openai.types.chat.chat_completion import ChatCompletion as Completion

from dataclasses import dataclass, field, asdict
from typing import List, Union, Dict

import logging
logger = logging.getLogger("autoplc_st")

@dataclass
class TokenUsage:
    """
    Tracks token usage statistics for input and output tokens
    
    Attributes:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
    """
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used (input + output)"""
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int):
        """Add input and output token counts to running totals"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

@dataclass
class TaskStatistics:
    """
    Statistics for a specific task, tracking token usage per role
    
    Attributes:
        task_name: Name of the task
        role_stats: Dictionary mapping role names to their token usage
        task_total: Total token usage across all roles for this task
    """
    task_name: str
    role_stats: Dict[str, TokenUsage] = field(default_factory=dict)
    task_total: TokenUsage = field(default_factory=TokenUsage)

@dataclass
class ExperimentStatistics:
    """
    Top-level statistics tracking token usage across all tasks
    
    Attributes:
        tasks: Dictionary mapping task names to their statistics
        total: Total token usage across all tasks
    """
    tasks: Dict[str, TaskStatistics] = field(default_factory=dict)
    total: TokenUsage = field(default_factory=TokenUsage)

    def get_or_create_task(self, task_name: str) -> TaskStatistics:
        """Get existing task statistics or create new ones if not found"""
        if task_name not in self.tasks:
            self.tasks[task_name] = TaskStatistics(task_name)
        return self.tasks[task_name]

    def add_usage(self, task_name: str, role_name: str, input_tokens: int, output_tokens: int):
        """
        Add token usage statistics for a specific task and role
        
        Updates token counts at task, role and total experiment levels
        """
        task_stat = self.get_or_create_task(task_name)
        if role_name not in task_stat.role_stats:
            task_stat.role_stats[role_name] = TokenUsage()
        task_stat.role_stats[role_name].add(input_tokens, output_tokens)
        task_stat.task_total.add(input_tokens, output_tokens)
        self.total.add(input_tokens, output_tokens)

    def to_dict(self):
        """Convert statistics to dictionary format"""
        return asdict(self)

retrieve_client = OpenAI(
    api_key=os.getenv("API_KEY_KNOWLEDGE").strip(),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

autoplc_client_anthropic = Anthropic(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

autoplc_client_openai = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

class _BaseClient:
    """
    用于与LLM（大语言模型）交互的基类。
    
    Attributes:
        config (Config): 本项目的配置类，包含了与LLM交互所需的配置信息。
    """
    def __init__(self, config: Config):
        self.config = config # 本项目的配置类
        self.experiment_base_logs_folder = 'default_logs' # 本次实验的日志文件夹，在init时重制
        self.statistics = ExperimentStatistics()

    def call(self, messages: str, task_name: str = "default", role_name: str = "default"):
        # 调用LLM的接口，发送消息并获取响应
        pass

    def save_statistics(self, 
            response: Completion, 
            task_name: str, 
            role_name: str = "default", 
            client_name: str = "default"
        ):
        """
        保存使用统计信息。
    
        此函数负责将给定响应对象中的使用信息（如输入和输出令牌数量）记录到统计系统中，并将这些信息保存到指定的JSON文件中。
    
        参数:
        - response: Completion或Message对象，包含响应数据，特别是使用信息。
        - task_name: str，任务名称，用于标识统计信息关联的任务。
        - role_name: str，默认为"default"，角色名称，用于进一步分类统计信息。
        - client_name: str，默认为"default"，客户端名称，用于组织统计信息的存储位置。
        """

        # 将使用信息添加到统计系统中    
        # print(response)
        # input_tokens = response.usage.prompt_tokens
        # output_tokens = response.usage.completion_tokens
        input_tokens = 1
        output_tokens = 1

        self.statistics.add_usage(
            task_name=task_name,
            role_name=role_name,
            input_tokens=input_tokens if input_tokens is not None else -1,
            output_tokens=output_tokens if output_tokens is not None else -1
        )
    
        # 构建日志文件的路径
        log_path = os.path.join(
            self.config.LOG_DIR, 
            self.experiment_base_logs_folder,
            client_name, 
            "statistics.json"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
        # 将统计信息序列化并保存到日志文件中
        with open(log_path, "w", encoding="utf-8") as log_file:
            json.dump(self._to_serializable(), log_file, indent=4)

    def _to_serializable(self):
        from dataclasses import asdict
        return asdict(self.statistics)

    @classmethod
    def extract_content(cls,response : Completion) -> str:
        try:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                content = response.choices[0].message.content
            else:
                content = response.choices[0]['message']['content']
        except (IndexError, KeyError, AttributeError):
            logger.error("无法从响应中提取内容，请检查响应结构。")
            content = None        
        return content

    def save_log(self, 
        messages: List[dict], 
        response: Completion, 
        task_name: str = "default", 
        role_name: str = "default"
    ):
        """
        保存消息和响应到日志文件。
        Args:
            messages (List[dict]): 消息列表，每个消息是一个字典。
            response: LLM的响应对象，包含了生成的消息和使用信息。
            task_name (str, optional): 任务名称，用于日志记录。
            role_name (str, optional): 角色名称，用于日志记录。
        """
        # 构造日志文件路径
        log_path = os.path.join(self.config.LOG_DIR, self.experiment_base_logs_folder, task_name, f"{role_name}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 将响应添加到消息列表中

        content = self.extract_content(response)

        messages_with_response = messages + [{"role": "assistant", "content": content}]

        # 从统计 POJO 中提取使用信息
        usage = self.statistics.tasks[task_name].role_stats.get(role_name, None)

        # 写入日志
        with open(log_path, "w", encoding="utf8") as f:
            f.write(PromptResultUtil.message_to_file_str(messages_with_response))
            if usage:
                f.write(
                    f"Token Usage:\nInput {usage.input_tokens} + Output {usage.output_tokens} = Total {usage.total_tokens}"
                )
            else:
                f.write("Token Usage: N/A\n")

class OpenAIClient(_BaseClient):
    """
    用于与LLM（大语言模型）交互的客户端类。
    
    Attributes:
        config (Config): 本项目的配置类，包含了与LLM交互所需的配置信息。
        experiment_base_logs_folder (str): 本次实验的日志文件夹，在init时重制。
        statistics ({}): 统计每个任务的input和output token的使用情况。
    """
    def __init__(self, config: Config):
        super().__init__(config)

    def call(self, 
        messages: List[dict], 
        task_name: str = "default", 
        role_name: str = "default",
        model: str = "gpt-4.1"
    ) -> Completion:
        """
        调用LLM的接口，发送消息并获取响应。
        
        Args:
            messages (List[dict]): 消息列表，每个消息是一个字典，包含role和content两个键。
            task_name (str, optional): 任务名称，用于日志记录。
            role_name (str, optional): 角色名称，用于日志记录。
        
        Returns:
            response: LLM的响应对象，包含了生成的消息和使用信息。
        """
        
        # 创建消息以获取响应
        response = autoplc_client_openai.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=False
        )
        # print(response)
        self.save_statistics(response, task_name, role_name=role_name, client_name="llm")
        self.save_log(messages, response, task_name, role_name)

        # 返回LLM的响应
        return response


class ZhipuAIQAClient(_BaseClient):
    """
    用于处理检索任务的客户端类，继承自_BaseClient。
    
    初始化时，接受一个Config对象作为配置参数。
    """
    def __init__(self, config: Config):
        # 初始化基类_BaseClient，传入配置参数
        super().__init__(config)

    def call_kbq(self, 
        messages: str, 
        task_name: str, 
        qa_prompt: str,
        knowledge_id : str,
        role_name: str = "retrieval"
    ) -> Completion:
        """
        该方法用于调用ZhipuAI的知识库问答接口，发送消息并获取响应。

        参数:
        - messages (str): 消息内容，通常是用户的问题。
        - task_name (str): 任务名称，用于日志记录。
        - qa_prompt (str): 问答提示模板，用于生成问答内容。
        - knowledge_id (str): 知识库ID，用于检索相关信息。
        - role_name (str, optional): 角色名称，用于日志记录，默认为"retrieval"。

        返回:
        - response: ZhipuAI的响应对象，包含生成的消息和使用信息。
        """

        # 创建模型完成请求，包含特定的检索工具参数
        response = retrieve_client.chat.completions.create(
            model=self.config.retrieve_model,  
            temperature=self.config.retrieve_temperature,
            top_p=self.config.retrieve_top_p,
            messages=messages,
            tools=[{
                "type": "retrieval",
                "retrieval": {
                    "knowledge_id": knowledge_id, 
                    "prompt_template": qa_prompt
                }
            }]
        )

        self.save_statistics(response, task_name, role_name = role_name, client_name="retrieval")
        self.save_log(messages, response, task_name, role_name)

        # 返回模型响应
        return response

def read_jsonl(filename):
    """Reads a jsonl file and yields each line as a dictionary"""
    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lines.append(json.loads(line))
    return lines

class BM25RetrievalInstruction:
    """
        A class that implements BM25-based instruction retrieval for PLC programming.
        This class provides functionality to search and retrieve relevant PLC instructions
        based on semantic similarity using the BM25 algorithm. It supports searching both
        by algorithm descriptions and direct queries.
        Attributes:
            instruction_corpus (List[str]): List of instruction documents containing API descriptions
            instruction_names (List[str]): List of instruction/API names
            tokenized_corpus (List[List[str]]): The tokenized version of instruction_corpus
            bm25 (BM25Okapi): The BM25 model used for retrieval
            INSTRUCTION_SCORE_THRESHOLD (float): Minimum score threshold for matching instructions
            INSTRUCTION_TOP_K (int): Maximum number of top results to return
        Parameters:
            config (Config): Configuration object containing necessary parameters:
                - INSTRUCTION_PATH: Path to instruction descriptions file
                - BM25_MODEL: Type of BM25 model to use (currently only supports "BM25Okapi")
                - INSTRUCTION_SCORE_THRESHOLD: Threshold score for instruction matching
                - INSTRUCTION_TOP_K: Number of top results to return
        Example:
            >>> config = Config()
            >>> retriever = _BM25RetrieverInstruction(config)
            >>> apis = retriever.query_algo_apis("首先，需要对输入的数组进行排序\n随后，设置计时器") 
            >>> docs = retriever.query_doc("如何使用计时器？")
    """
    def __init__(self, config: Config):
        jieba.initialize()
        stop_path = config.INSTRUCTION_DIR.joinpath("stopwords_english.txt.")
        stopwords = set(open(stop_path, encoding="utf-8").read().splitlines())
        self.stopwords = stopwords
        # 初始化API描述和名称
        self.instruction_corpus,self.instruction_names = self.read_ins_desc(config.INSTRUCTION_PATH)

        logger.info(f"loading instruction from >>> {config.INSTRUCTION_PATH}")

        def remove_stopwords(token_list, stopwords):
            return [t for t in token_list if t not in stopwords]

        self.tokenized_corpus = [
            remove_stopwords(list(jieba.cut(doc)), stopwords)
            for doc in self.instruction_corpus
        ]
        
        # 初始化BM25模型
        if config.BM25_MODEL == "BM25Okapi":
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            raise NotImplementedError(f"BM25 model {config.BM25_MODEL} not implemented")
        
        # 设置BM25模型的参数
        self.INSTRUCTION_SCORE_THRESHOLD = config.INSTRUCTION_SCORE_THRESHOLD
        self.INSTRUCTION_TOP_K = config.INSTRUCTION_TOP_K

    def query_api_by_type(self , complex_types:list[str]) -> list[str]:
        """
        返回一组api，其中的输入参数为 complex_types 中的类型
        """
        pass

    def query_algo_apis(self, algo: str) -> List[str]:
        """
        Returns a list of APIs related to the algorithm query string provided.

        This method processes the query string by tokenization and BM25 algorithm to find APIs related to the algorithm.
        It first splits the query string into separate query lines, then processes each line, 
        calculates the matching score between the query and the document using the BM25 algorithm,
        and selects the most relevant API documents based on the score sorting.

        Parameters:
        algo (str): The algorithm query string entered by the user, which can contain multiple lines of query.

        Returns:
        List[str]: A list of API names related to the query algorithm, where each str is an API name.
        """
        # 初始化一个集合，用于存储所有相关的API名称，以避免重复
        all_apis = set()
        
        # 将算法查询字符串按行分割，每行作为一个独立的查询
        algo_lines = algo.strip().split("\n") 

        for line in algo_lines:
            if line:
                
                # 使用jieba分词将查询行分割成单词列表
                tokenized_query = list(jieba.cut(line))

                # 去除停用词
                tokenized_query = [t for t in tokenized_query if t not in self.stopwords]

                # 使用BM25算法获取当前查询与所有文档的得分
                # print(self)
                # print("----------")
                # print(tokenized_query)

                scores = self.bm25.get_scores(tokenized_query)
                ranked_indices = np.argsort(scores)[::-1]
                ranked_scores = sorted(scores, reverse=True)

                # 过滤出得分高于预设阈值的文档，并限制结果数量
                top_docs = [(rank, score) for rank, score in zip(ranked_indices, ranked_scores) if score > self.INSTRUCTION_SCORE_THRESHOLD][:self.INSTRUCTION_TOP_K]
                for idx, (doc_idx, score) in enumerate(top_docs, start=1):
                    # 将相关API添加到集合中
                    all_apis.add(self.instruction_names[doc_idx])
    
        all_apis = list(all_apis)
        return all_apis

    def query_doc(self, query, top_n=3) -> List[str]:
        """
        Based on the user's query, return a list of documents most relevant to the question.

        Parameters:
        - query (str): The user's query.
        - top_n (int): The number of most relevant documents returned, default is 3.

        Returns:
        - List[str]: The list of documents most relevant to the query.
        """
        # 使用jieba分词器对查询问题进行分词
        tokenized_query = list(jieba.cut(query))
        
        # 使用BM25算法获取与查询问题最相关的文档
        return self.bm25.get_top_n(tokenized_query, self.instruction_corpus, n=top_n)
    
    def read_ins_desc(self, inst_paths: List[str]):
        inst_desc = []
        for inst_path in inst_paths:
            inst_desc.extend(read_jsonl(inst_path))
        ret = []
        names = []
        for api in inst_desc:
            # 提取 description 字段并转为小写
            description = api['description'].lower()
            # 使用 jieba 分词
            tokenized_desc = list(jieba.cut(description))
            # 去除停用词
            filtered_desc = [t for t in tokenized_desc if t not in self.stopwords]
            # 将处理后的结果添加到 ret 列表
            ret.append(" ".join(filtered_desc))
            names.append(api['instruction_name'])
        return ret, names

class ClientManager:
    """
    A Singleton class that manages the initialization and retrieval of various clients used for AI and API interactions.
    It ensures that only one instance of each client is created and provides access to them globally.
    """
    _instance = None
    _openai_client : OpenAIClient = None
    _zhipuai_client : ZhipuAIQAClient = None
    _local_api_retriever : BM25RetrievalInstruction = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientManager, cls).__new__(cls)
        return cls._instance

    def set_config(self, config: Config):
        self._openai_client = OpenAIClient(config)
        self._zhipuai_client = ZhipuAIQAClient(config)
        self._local_api_retriever = BM25RetrievalInstruction(config)

    def get_openai_client(self):
        if self._openai_client is None:
            raise Exception("OpenAIClient未初始化")
        return self._openai_client

    def get_zhipuai_client(self):
        if self._zhipuai_client is None:
            raise Exception("ZhipuAIQAClient未初始化")
        return self._zhipuai_client

    def get_local_api_retriever(self):
        if self._local_api_retriever is None:
            raise Exception("BM25RetrievalInstruction未初始化")
        return self._local_api_retriever