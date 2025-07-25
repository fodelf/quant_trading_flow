from crewai import Agent, Crew, Process, Task
import os
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from quant_trading_flow.modules.deepseek import deepseek_llm
from quant_trading_flow.crews.cfo.tools import cfo_tool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class CfoCrew:
    """CFO Crew"""

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
    def cfo(self) -> Agent:
        return Agent(
            config=self.agents_config["cfo"],  # type: ignore[index]
            llm=deepseek_llm,
            max_retry_limit=10,
            max_execution_time=1800,
            tools=[
                cfo_tool.get_data_report,
                cfo_tool.get_data_analysis,
                cfo_tool.get_government_affairs,
                cfo_tool.get_public_sentiment,
                cfo_tool.get_strategy_report,
                cfo_tool.get_risk_report,
            ],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def cfo_data_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_data_analysis_task"],  # type: ignore[index]
        )

    @task
    def cfo_fundamental_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_fundamental_analysis_task"],  # type: ignore[index]
        )

    @task
    def cfo_policy_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_policy_analysis_task"],  # type: ignore[index]
        )

    @task
    def cfo_sentiment_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_sentiment_analysis_task"],  # type: ignore[index]
        )

    @task
    def cfo_strategy_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_strategy_analysis_task"],  # type: ignore[index]
        )

    @task
    def cfo_risk_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_risk_analysis_task"],  # type: ignore[index]
        )

    @task
    def cfo_task(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_task"],
            context=[self.cfo_data_analysis_task(), self.cfo_fundamental_analysis_task(), self.cfo_policy_analysis_task(), self.cfo_sentiment_analysis_task(), self.cfo_strategy_analysis_task(), self.cfo_risk_analysis_task()],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Cfo Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge
        custom_storage_path = "./storage"
        os.makedirs(custom_storage_path, exist_ok=True)
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            planning=True,
            # memory=True,
            planning_llm=deepseek_llm,
            # long_term_memory=LongTermMemory(
            #     storage=LTMSQLiteStorage(db_path=f"{custom_storage_path}/memory.db")
            # ),
        )
