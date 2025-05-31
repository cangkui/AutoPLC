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
from autoplc_scl.tools import PromptResultUtil
from anthropic.types.message import Message
# from zhipuai.types.chat.chat_completion import Completion
from openai.types.chat.chat_completion import ChatCompletion as Completion

from dataclasses import dataclass, field, asdict
from typing import List, Union, Dict

import logging
logger = logging.getLogger("autoplc_scl")

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

class _BaseClient:
    """
    Base class for interacting with LLM
    
    Attributes:
        config (Config): The configuration class of this project contains the configuration information required to interact with LLM.
    """
    def __init__(self, config: Config, llm_client: Union[OpenAI, Anthropic]):
        self.config = config # Configuration class of this project
        self.experiment_base_logs_folder = 'default_logs' # The log folder of this experiment is remade during init
        self.statistics = ExperimentStatistics()
        self.llm_client = llm_client # LLM client instance

    def call(self, messages: str, task_name: str = "default", role_name: str = "default"):
        # Call the LLM interface, send a message and get a response
        pass

    def save_statistics(self, 
            response: Completion, 
            task_name: str, 
            role_name: str = "default", 
            client_name: str = "default"
        ):
        """
        Save usage statistics.
    
        This function is responsible for recording usage information in a given response object (such as the number of input and output tokens) to the statistical system and saving this information to the specified JSON file.
    
        parameter:
        -response: Completion or Message object, containing response data, especially usage information.
        -task_name: str, task name, used to identify the task associated with statistics.
        -role_name: str, default is "default", role name, used for further classification statistics.
        -client_name: str, default is "default", client name, storage location for organization statistics.
        """

        # Add usage information to the statistics system
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
    
        # The path to build the log file
        log_path = os.path.join(
            self.config.LOG_DIR, 
            self.experiment_base_logs_folder,
            client_name, 
            "statistics.json"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
        # Serialize and save statistics to log file
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
            logger.error("Unable to extract content from the response, please check the response structure.")
            content = None        
        return content

    def save_log(self, 
        messages: List[dict], 
        response: Completion, 
        task_name: str = "default", 
        role_name: str = "default"
    ):
        """
        Save messages and responses to log files.
        Args:
            messages (List[dict]): List of messages, each message is a dictionary.
            response: LLM's response object, containing the generated message and usage information.
            task_name (str, optional): task name, used for logging.
            role_name (str, optional): role name, used for logging.
        """
        # Construct log file path
        log_path = os.path.join(self.config.LOG_DIR, self.experiment_base_logs_folder, task_name, f"{role_name}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Add a response to the message list

        content = self.extract_content(response)

        messages_with_response = messages + [{"role": "assistant", "content": content}]

        # Extract usage information from statistical POJO
        usage = self.statistics.tasks[task_name].role_stats.get(role_name, None)

        # Write to log
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
    A client class for interacting with LLM (large language model).
    
    Attributes:
        config (Config): The configuration class of this project contains the configuration information required to interact with LLM.
        experiment_base_logs_folder (str): The log folder of this experiment is remade during init.
        statistics ({}): Statistics the usage of input and output tokens for each task.
    """
    def __init__(self, config: Config, llm_client: Union[OpenAI, Anthropic]):
        super().__init__(config, llm_client)

    def call(self, 
        messages: List[dict], 
        task_name: str = "default", 
        role_name: str = "default",
        model: str = None
    ) -> Completion:
        """
        Call the LLM's interface, send a message and get a response.
        
        Args:
            messages (List[dict]): message list, each message is a dictionary, containing role and content keys.
            task_name (str, optional): task name, used for logging.
            role_name (str, optional): role name, used for logging.
        
        Returns:
            response: LLM's response object, containing the generated message and usage information.
        """
        if model is None:
            model = self.config.model

        response = self.llm_client.chat.completions.create(
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

        return response


class ZhipuAIQAClient(_BaseClient):
    """
    The client class used to handle retrieval tasks, inherited from _BaseClient.
    
    During initialization, a Config object is accepted as a configuration parameter.
    """
    def __init__(self, config: Config,llm_client: Union[OpenAI, Anthropic]):
        # Initialize the base class_BaseClient, pass in the configuration parameters
        super().__init__(config, llm_client)

    def call_kbq(self, 
        messages: str, 
        task_name: str, 
        qa_prompt: str,
        knowledge_id : str,
        role_name: str = "retrieval"
    ) -> Completion:
        """
        This method is used to call ZhipuAI's knowledge base question and answer interface, send messages and get responses.

        parameter:
        -messages (str): Message content, usually a user's problem.
        -task_name (str): Task name, used for logging.
        -qa_prompt (str): Q&A prompt template, used to generate Q&A content.
        -knowledge_id (str): Knowledge Base ID, used to retrieve relevant information.
        -role_name (str, optional): Role name, used for logging, default to "retrieval".

        return:
        -response: ZhipuAI's response object, containing generated messages and usage information.
        """

        # Create a model completion request, including specific retrieval tool parameters
        response = self.llm_client.chat.completions.create(
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

        return response

def read_jsonl(filename):
    """Reads a jsonl file and yields each line as a dictionary"""
    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lines.append(json.loads(line))
    return lines

class BM25RetrievalInstruction:
    def __init__(self, config:Config):
        jieba.initialize()
        stop_path = config.INSTRUCTION_DIR.joinpath("stopwords_english.txt.")
        self.stopwords = set(open(stop_path, encoding="utf-8").read().splitlines())

        # Load instruction data
        self.instructions = []
        for inst_path in config.INSTRUCTION_PATH:
            self.instructions.extend(read_jsonl(inst_path))
        self.instruction_names = [api['instruction_name'] for api in self.instructions]

        # Multi-channel text construction
        self.channel_texts = {
            'keywords': [],
            'summary': [],
            'usage': []
        }

        logger.info(f"initializing BM25s with chanel_texts: {self.channel_texts.keys()}")

        for api in self.instructions:
            self.channel_texts['keywords'].append(self._tokenize(api['generated_keywords']))
            self.channel_texts['summary'].append(self._tokenize(api['generated_brief']['functional_summary']))
            self.channel_texts['usage'].append(self._tokenize(api['generated_brief']['usage_context']))
        
        # Initialize the BM25 model
        self.bm25_models = {
            'keywords': BM25Okapi(self.channel_texts['keywords']),
            'summary': BM25Okapi(self.channel_texts['summary']),
            'usage': BM25Okapi(self.channel_texts['usage'])
        }

        self.INSTRUCTION_SCORE_THRESHOLD = config.INSTRUCTION_SCORE_THRESHOLD
        self.INSTRUCTION_TOP_K = config.INSTRUCTION_TOP_K

    def _tokenize(self, content:str) -> List[str]:
        """Tokenizes the input content using jieba and removes stopwords."""
        if isinstance(content, list):
            content = ".".join(content)
        tokens = list(jieba.cut(content.lower()))
        return [t for t in tokens if t not in self.stopwords]

    def query_multi_channel(self, query: str) -> List[str]:
        """
        Queries multiple channels (keywords, summary, usage) to find relevant instructions.
        Args:
            query (str): The query string to search for.
        Returns:
            List[str]: A list of instruction names that match the query.
        """
        import re

        def split_text(text: str) -> List[str]:
            return [line.strip() for line in re.split(r'[；;。\n]+', text) if line.strip()]

        matched_apis = set()

        for sentence in split_text(query):
            tokenized = self._tokenize(sentence)
            for channel, bm25 in self.bm25_models.items():
                scores = bm25.get_scores(tokenized)
                scored_items = [
                    (self.instruction_names[i], score)
                    for i, score in enumerate(scores)
                    if score > self.INSTRUCTION_SCORE_THRESHOLD
                ]
                top_hits = sorted(scored_items, key=lambda x: x[1], reverse=True)[:self.INSTRUCTION_TOP_K]
                matched_apis.update([name for name, _ in top_hits])

        return list(matched_apis)

    def query_api_by_type(self, complex_types: List[str]) -> List[str]:
        """
        Queries the API instructions based on complex types.
        args:
            complex_types (List[str]): A list of complex types to search for.
        Returns:
            List[str]: A list of instruction names that match the complex types.
        """
        matched_apis = set()
        norm_types = [t.lower().replace('[*]', '').replace('_', '') for t in complex_types]
        for name, api in zip(self.instruction_names, self.instructions):
            all_text = json.dumps(api).lower().replace('_', '')
            if any(t in all_text for t in norm_types):
                matched_apis.add(name)
        return list(matched_apis)

    def query_doc(self, query: str, channel: str = 'summary', top_n: int = 3) -> List[str]:
        """
        Queries the BM25 model for the top N instructions based on the query.
        Args:
            query (str): The query string to search for.
            channel (str): The channel to search in ('keywords', 'summary', 'usage').
            top_n (int): The number of top results to return.
        Returns:
                List[str]: A list of instruction names that match the query.
        """
        tokenized = self._tokenize(query)
        bm25 = self.bm25_models[channel]
        return bm25.get_top_n(tokenized, self.instruction_names, n=top_n)

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

        self._openai_client = OpenAIClient(config,autoplc_client_openai)
        self._zhipuai_client = ZhipuAIQAClient(config,retrieve_client)
        self._local_api_retriever = BM25RetrievalInstruction(config)

    def get_openai_client(self):
        if self._openai_client is None:
            raise Exception("OpenAIClient Not initialized")
        return self._openai_client

    def get_zhipuai_client(self):
        if self._zhipuai_client is None:
            raise Exception("ZhipuAIQAClient Not initialized")
        return self._zhipuai_client

    def get_local_api_retriever(self):
        if self._local_api_retriever is None:
            raise Exception("BM25RetrievalInstruction Not initialized")
        return self._local_api_retriever