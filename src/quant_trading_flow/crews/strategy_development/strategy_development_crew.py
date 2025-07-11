from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from quant_trading_flow.modules.deepseek import deepseek_llm
from quant_trading_flow.crews.strategy_development.tools import (
    strategy_development_tool,
)
from quant_trading_flow.crews.strategy_development.tools import (
    strategy_development_report,
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class StrategyDevelopmentCrew:
    """StrategyDevelopment Crew"""

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
    def strategy_development(self) -> Agent:
        return Agent(
            config=self.agents_config["strategy_development"],  # type: ignore[index]
            verbose=True,
            llm=deepseek_llm,
            max_retry_limit=5,
            max_execution_time=600,
            tools=[
                strategy_development_report.get_data_report,
                # strategy_development_report.get_data_analysis,
                strategy_development_tool.optimize_for_high_return,
            ],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["strategy_task"],  # type: ignore[index]
            # output_file='output/strategy_report.md'
            # output_file="output/strategy_report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the StrategyDevelopment Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            llm=deepseek_llm,
        )
