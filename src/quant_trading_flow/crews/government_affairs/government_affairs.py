from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from quant_trading_flow.modules.deepseek import deepseek_llm


@CrewBase
class GovernmentAffairsCrew:
    """GovernmentAffairs Crew"""

    """获取宏观政策与市场环境----"""
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
    def government_affairs(self) -> Agent:
        return Agent(
            config=self.agents_config["government_affairs"],  # type: ignore[index]
            verbose=True,
            max_retry_limit=5,
            max_execution_time=600,
            llm=deepseek_llm,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def government_affairs_task(self) -> Task:
        return Task(
            config=self.tasks_config["government_affairs_task"],  # type: ignore[index]
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
            llm=deepseek_llm,
        )
