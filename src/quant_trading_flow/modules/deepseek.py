import os
from crewai import LLM
from langchain_openai import ChatOpenAI

# from langchain.chat_models import ChatOpenAI
from openai import OpenAI

deepseek_llm = LLM(
    model="deepseek/deepseek-reasoner",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

openai_llm = LLM(
    model="openrouter/openai/gpt-5-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)
