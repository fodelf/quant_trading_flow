from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from quant_trading_flow.modules.deepseek import deepseek_llm
from quant_trading_flow.crews.fundamental_analysis.tools import data_base


@CrewBase
class FundamentalAnalysisCrew:
    """FundamentalAnalysis Crew"""

    """获取基本面（公司经营状况，行业前景，财务指标，资产负债表，利润表，现金流量表）"""
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
    def data_analysis(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analysis"],  # type: ignore[index]
            verbose=True,
            llm=deepseek_llm,
            max_retry_limit=3,
            tools=[data_base.get_finance_data_str],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def data_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_analysis_task"],  # type: ignore[index]
            # output_file="output/data_analysis.md",
            # output_file='output/data_report.html'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FundamentalAnalysis Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            llm=deepseek_llm,
        )
