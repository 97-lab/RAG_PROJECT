from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import DEEPSEEK_API_KEY

llm = ChatOpenAI(  # deepseek 的（走 OpenAI 兼容接口）
    temperature=0,
    model='deepseek-chat',
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com")


web_search_tool = TavilySearchResults(max_results=2)

# 原来的 openai（xiaoai.plus 代理）配置，保留备用
# llm = ChatOpenAI(
#     temperature=0,
#     model='gpt-4o-mini',
#     api_key=OPENAI_API_KEY,
#     base_url="https://xiaoai.plus/v1")