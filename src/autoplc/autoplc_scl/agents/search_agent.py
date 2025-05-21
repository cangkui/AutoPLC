import logging
from typing import List, Dict
import wikipedia
from googlesearch import search
from bs4 import BeautifulSoup
import requests
from autoplc_scl.agents.clients import OpenAIClient

logger = logging.getLogger("autoplc_scl")

class KnowledgeAugmentAgent:
    """
    根据给定的SCL任务，识别不熟悉的知识点，并分别从Wiki与Google抓取内容，通过大模型总结后提供给用户。

    Methods:
        run(task_desc: str, openai_client: OpenAIClient) -> Dict[str, str]
            执行知识增强流程。
    """

    @classmethod
    def run(cls, task_desc: str, openai_client: OpenAIClient) -> Dict[str, str]:
        """主入口"""
        logger.info("Start extracting unknown knowledge points...")
        concept_groups = cls.extract_knowledge_points(task_desc, openai_client)

        logger.info("Start querying knowledge...")
        results = {}
        for item in concept_groups["scl_elements"]:
            content = cls.query_google(item)
            if content:
                summary = cls.summarize(content, item, task_desc, openai_client)
                results[item] = summary
            else:
                results[item] = "未获取到相关信息。"

        for item in concept_groups["general_elements"]:
            content = cls.query_wiki(item)
            if content:
                summary = cls.summarize(content, item, task_desc, openai_client)
                results[item] = summary
            else:
                results[item] = "未获取到相关信息。"

        return results

    @classmethod
    def extract_knowledge_points(cls, task: str, client: OpenAIClient) -> Dict[str, List[str]]:
        prompt = f"""
你是一个PLC工程师助手，面对以下SCL编程任务：

"{task}"

请你分析自己不熟悉的知识点，列出需要进一步查阅的概念，分为两类：
1. SCL相关的专业概念（如数据类型、PLC指令等）
2. 通用编程或工程概念（如算法、时间标准等）

返回格式为JSON:
{{
  "scl_elements": [...],
  "general_elements": [...]
}}
"""
        messages = [{"role": "user", "content": prompt}]
        resp = client.call(messages, task_name="KnowledgeExtract", role_name="knowledge")
        return eval(resp.choices[0].message.content)

    @classmethod
    def query_google(cls, query: str) -> str:
        try:
            url = list(search(f"{query} site:www.ad.siemens.com.cn", num_results=1))[0]
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator="\n")[:4000]
        except Exception as e:
            logger.warning(f"Google 查询失败: {e}")
            return ""

    @classmethod
    def query_wiki(cls, query: str, lang: str = 'zh') -> str:
        try:
            wikipedia.set_lang(lang)
            page = wikipedia.page(query)
            return page.content[:4000]
        except Exception as e:
            logger.warning(f"Wiki 查询失败: {e}")
            return ""

    @classmethod
    def summarize(cls, raw_text: str, concept: str, task: str, client: OpenAIClient) -> str:
        prompt = f"""
你是一名SCL编程专家。以下是关于“{concept}”的内容，请结合任务：\n"{task}"\n，总结该知识点与任务相关的核心信息、用法或注意事项，忽略无关内容：

\"\"\"{raw_text}\"\"\"
"""
        messages = [{"role": "user", "content": prompt}]
        resp = client.call(messages, task_name="KnowledgeSummary", role_name="knowledge")
        return resp.choices[0].message.content
