import os
from crewai import LLM
deepseek_llm = LLM(
    model="deepseek/deepseek-reasoner",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)
