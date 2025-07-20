from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
import os

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from quant_trading_flow.modules.deepseek import deepseek_llm
from quant_trading_flow.crews.stock_screener.tools.stock_screener_tool import (
    get_filtered_stocks,
)


@CrewBase
class StockScreenerCrew:
    """StockScreener Crew"""

    """选股"""
    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def stock_screener(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_screener"],  # type: ignore[index]
            verbose=True,
            max_retry_limit=5,
            max_execution_time=1800,
            llm=deepseek_llm,
            tools=[get_filtered_stocks],
            # system_template="""你是 {role}。 {backstory}。
            #   你的目标是: {goal}。
            #   按照任务的顺序依次执行。
            #   以自然和对话的方式回应。专注于提供有用、准确的信息。""",
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def stock_screener_pick_task(self) -> Task:
        return Task(
            config=self.tasks_config["stock_screener_pick_task"],  # type: ignore[index]
        )

    @task
    def stock_screener_task(self) -> Task:
        return Task(
            config=self.tasks_config["stock_screener_task"],  # type: ignore[index]
            context=[self.stock_screener_pick_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the StockScreener Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge
        custom_storage_path = "./storage/stock"
        os.makedirs(custom_storage_path, exist_ok=True)
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            planning=True,
            planning_llm=deepseek_llm,
            # memory=True,
            # long_term_memory=LongTermMemory(
            #     storage=LTMSQLiteStorage(db_path=f"{custom_storage_path}/memory.db")
            # ),
        )
