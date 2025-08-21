from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from quant_trading_flow.modules.deepseek import deepseek_llm, openai_llm
from crewai_tools import WebsiteSearchTool


@CrewBase
class PublicSentimentCrew:
    """PublicSentiment Crew"""

    """获取舆情（新闻，社交媒体，论坛）----"""
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
    def public_sentiment(self) -> Agent:
        return Agent(
            config=self.agents_config["public_sentiment"],  # type: ignore[index]
            verbose=True,
            max_retry_limit=5,
            max_execution_time=1800,
            llm=openai_llm,
            # tools=[WebsiteSearchTool],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def public_sentiment_task(self) -> Task:
        return Task(
            config=self.tasks_config["public_sentiment_task"],  # type: ignore[index]
            # output_file="output/public_sentiment.md",
            # output_file='output/data_report.html'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PublicSentiment Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # llm=deepseek_llm,
        )
