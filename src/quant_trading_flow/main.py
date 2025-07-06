#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from datetime import datetime

from quant_trading_flow.crews.data_engineer.data_engineer import DataEngineerCrew
from quant_trading_flow.crews.strategy_development.strategy_development_crew import (
    StrategyDevelopmentCrew,
)
from quant_trading_flow.crews.risk_management.risk_management import RiskManagementCrew
from quant_trading_flow.crews.cfo.cfo import CfoCrew


class TradingState(BaseModel):
    symbol: str = "SHA:600513"
    start_date: str = "20180101"
    end_date: str = datetime.now().strftime("%Y%m%d")
    file_date: str = datetime.now().strftime("%Y%m%d%H%M%S")


class TradingFlow(Flow[TradingState]):

    @start()
    def fetch_market_data(self):
        # 获取目标市场数据
        return {
            # "symbol": "SHE:002036",
            # "symbol": "SHA:688551",
            "symbol": self.state.symbol,
            "start_date": self.state.start_date,
            "end_date": self.state.end_date,
            "file_date": self.state.file_date,
        }

    @listen(fetch_market_data)
    def handle_data_with_crew(self, market_data):
        result = DataEngineerCrew().crew().kickoff(inputs=market_data)
        return result.raw

    @listen(handle_data_with_crew)
    def strategy_development_with_crew(self):
        result = (
            StrategyDevelopmentCrew()
            .crew()
            .kickoff(
                inputs={
                    "symbol": self.state.symbol,
                    "start_date": self.state.start_date,
                    "end_date": self.state.end_date,
                    "file_date": self.state.file_date,
                }
            )
        )
        return result.raw

    @listen(strategy_development_with_crew)
    def risk_management_with_crew(self):
        result = (
            RiskManagementCrew()
            .crew()
            .kickoff(
                inputs={
                    "symbol": self.state.symbol,
                    "start_date": self.state.start_date,
                    "end_date": self.state.end_date,
                    "file_date": self.state.file_date,
                }
            )
        )
        return result.raw

    @listen(risk_management_with_crew)
    def cfo_with_crew(self):
        result = (
            CfoCrew()
            .crew()
            .kickoff(
                inputs={
                    "symbol": self.state.symbol,
                    "start_date": self.state.start_date,
                    "end_date": self.state.end_date,
                    "file_date": self.state.file_date,
                }
            )
        )
        return result.raw


def kickoff():
    list = [
        {
            "symbol": "688551",
            "start_date": "20180101",
            "end_date": datetime.now().strftime("%Y%m%d"),
            "file_date": datetime.now().strftime("%Y%m%d%H%M%S"),
        },
    ]
    for item in list:
        TradingState.symbol = item["symbol"]
        TradingState.start_date = item["start_date"]
        TradingState.end_date = item["end_date"]
        trading_flow = TradingFlow()
        trading_flow.kickoff()


def plot():
    trading_flow = TradingFlow()
    trading_flow.plot()


if __name__ == "__main__":
    plot()
