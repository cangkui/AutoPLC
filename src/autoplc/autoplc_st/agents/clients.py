from openai import OpenAI
from zhipuai import ZhipuAI
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

retrieve_client = ZhipuAI(
    api_key=os.getenv("API_KEY_KNOWLEDGE")
)
# API_KEY="sk-zk24e98452f64e4b84e7eec2348aba232e067dfafce61520"

# autoplc_client_openai = OpenAI(
#     base_url=os.getenv("BASE_URL"),
#     api_key=os.getenv("API_KEY")
# )

autoplc_client_anthropic = Anthropic(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

BASE_MODEL: str = "claude-3-5-sonnet-20241022"
RAG_DATA_DIR = os.path.join(os.getenv("ROOTPATH"), "data/rag_data")