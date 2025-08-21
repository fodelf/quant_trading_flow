from crewai import Agent, Crew, Process, Task
import os
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from quant_trading_flow.modules.deepseek import deepseek_llm, openai_llm
from quant_trading_flow.crews.cfo.tools import cfo_tool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class CfoHasDownCrew:
    """CfoHasDown Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents_down.yaml"
    tasks_config = "config/tasks_has_down.yaml"

    @agent
    def cfo_down(self) -> Agent:
        return Agent(
            config=self.agents_config["cfo_down"],  # type: ignore[index]
            llm=deepseek_llm,
            max_retry_limit=10,
            max_execution_time=1800,
        )

    @task
    def cfo_task_has_down(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_task_has_down"],
        )

    @crew
    def crew(self) -> Crew:
        """CfoHasDown Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
