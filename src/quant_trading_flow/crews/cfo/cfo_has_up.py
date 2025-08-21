from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from quant_trading_flow.modules.deepseek import deepseek_llm


@CrewBase
class CfoHasUpCrew:
    """CfoHasUp  Crew"""

    # agents: List[BaseAgent]
    # tasks: List[Task]

    agents_config = "config/agents_up.yaml"
    tasks_config = "config/tasks_has_up.yaml"

    @agent
    def cfo_up(self) -> Agent:
        return Agent(
            config=self.agents_config["cfo_up"],  # type: ignore[index]
            llm=deepseek_llm,
            max_retry_limit=10,
            max_execution_time=1800,
        )

    @task
    def cfo_task_has_up(self) -> Task:
        return Task(
            config=self.tasks_config["cfo_task_has_up"],
        )

    @crew
    def crew(self) -> Crew:
        """CfoHasUp Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
