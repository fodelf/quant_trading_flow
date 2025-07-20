from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from quant_trading_flow.modules.deepseek import deepseek_llm
from quant_trading_flow.crews.data_engineer.tools import data_tool


@CrewBase
class DataEngineerCrew:
    """DataEngineer Crew"""

    """获取交易趋势（K线图，技术指标，成交量）----"""
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
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],  # type: ignore[index]
            verbose=True,
            llm=deepseek_llm,
            max_retry_limit=5,
            max_execution_time=1800,
            tools=[data_tool.get_china_stock_data],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def get_data_task(self) -> Task:
        return Task(
            config=self.tasks_config["get_data_task"],  # type: ignore[index]
        )

    @task
    def data_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_task"],  # type: ignore[index]
            context=[self.get_data_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Data Engineer Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            planning=True,
            planning_llm=deepseek_llm,
        )
