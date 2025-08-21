from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from quant_trading_flow.modules.deepseek import deepseek_llm, openai_llm
from quant_trading_flow.crews.risk_management.tools import risk_management_tool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class RiskHasManagementCrew:
    """RiskHasManagement Crew"""

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
    def risk_management(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_management"],  # type: ignore[index]
            llm=deepseek_llm,
            max_retry_limit=5,
            max_execution_time=1800,
            # tools=[
            #     risk_management_tool.get_data_report,
            #     risk_management_tool.get_data_analysis,
            #     risk_management_tool.get_government_affairs,
            #     risk_management_tool.get_public_sentiment,
            #     risk_management_tool.get_strategy_report,
            # ],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    # @task
    # def risk_management_data_analysis_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["risk_management_data_analysis_task"],  # type: ignore[index]
    #     )

    # @task
    # def risk_management_fundamental_analysis_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["risk_management_fundamental_analysis_task"],  # type: ignore[index]
    #     )

    # @task
    # def risk_management_policy_analysis_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["risk_management_policy_analysis_task"],  # type: ignore[index]
    #     )

    # @task
    # def risk_management_sentiment_analysis_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["risk_management_sentiment_analysis_task"],  # type: ignore[index]
    #     )

    # @task
    # def risk_management_strategy_analysis_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["risk_management_strategy_analysis_task"],  # type: ignore[index]
    #     )

    @task
    def risk_management_task(self) -> Task:
        return Task(
            config=self.tasks_config["risk_has_management_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """RiskHasManagement Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # planning=True,
            # planning_llm=deepseek_llm,
        )
