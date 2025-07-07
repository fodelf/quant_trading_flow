#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, and_, router

from datetime import datetime

from rich import console

import pandas as pd
import os

from quant_trading_flow.crews.data_engineer.data_engineer import DataEngineerCrew
from quant_trading_flow.crews.fundamental_analysis.fundamental_analysis import (
    FundamentalAnalysisCrew,
)
from quant_trading_flow.crews.government_affairs.government_affairs import (
    GovernmentAffairsCrew,
)
from quant_trading_flow.crews.public_sentiment.public_sentiment import (
    PublicSentimentCrew,
)

from quant_trading_flow.crews.strategy_development.strategy_development_crew import (
    StrategyDevelopmentCrew,
)
from quant_trading_flow.crews.risk_management.risk_management import RiskManagementCrew
from quant_trading_flow.crews.cfo.cfo import CfoCrew


def append_number_to_csv(file_path, number):
    """
    向CSV文件追加数字：如果文件不存在则创建并写入，如果存在则追加

    参数:
        file_path (str): CSV文件路径
        number (int/float): 要追加的数字
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 文件不存在，创建新文件并写入数字
        df = pd.DataFrame([number], columns=["Value"])
        df.to_csv(file_path, index=False)
        print(f"✅ 创建新文件 '{file_path}' 并写入初始值: {number}")
    else:
        try:
            # 文件存在，读取现有数据
            df = pd.read_csv(file_path)

            # 创建新行并追加
            new_row = pd.DataFrame([number], columns=["Value"])
            df = pd.concat([df, new_row], ignore_index=True)

            # 保存回文件
            df.to_csv(file_path, index=False)
            print(f"📝 在 '{file_path}' 中追加值: {number}")

        except Exception as e:
            print(f"❌ 追加数据失败: {e}")


class TradingState(BaseModel):
    symbol: str = "SHA:600513"
    start_date: str = "20180101"
    end_date: str = datetime.now().strftime("%Y%m%d")
    file_date: str = datetime.now().strftime("%Y%m%d%H%M%S")
    risk_approval_flag: bool = True
    trade_approval_flag: bool = True


class TradingFlow(Flow[TradingState]):

    @start()
    def init_market_data(self):
        # 获取目标市场数据
        return {
            # "symbol": "SHE:002036",
            # "symbol": "SHA:688551",
            "symbol": self.state.symbol,
            "start_date": self.state.start_date,
            "end_date": self.state.end_date,
            "file_date": self.state.file_date,
        }

    @listen(init_market_data)
    def handle_data(self, market_data):
        result = DataEngineerCrew().crew().kickoff(inputs=market_data)
        return result.raw

    @listen(init_market_data)
    def fundamental_analysis(self, market_data):
        result = FundamentalAnalysisCrew().crew().kickoff(inputs=market_data)
        return result.raw

    @listen(init_market_data)
    def government_affairs(self, market_data):
        result = GovernmentAffairsCrew().crew().kickoff(inputs=market_data)
        return result.raw

    @listen(init_market_data)
    def public_sentiment(self, market_data):
        result = PublicSentimentCrew().crew().kickoff(inputs=market_data)
        return result.raw

    @listen(handle_data)
    def strategy_development(self):
        print(self.state.file_date)
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

    @listen(
        and_(
            strategy_development,
            public_sentiment,
            fundamental_analysis,
            government_affairs,
        )
    )
    def risk_management(self):
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
        self.state.risk_approval_flag = "同意买入" in result.raw
        return result.raw

    @router(risk_management)
    def risk_approval(self):
        if self.state.risk_approval_flag:
            return "risk_approval_pass"
        else:
            return "approval_reject"

    @router("risk_approval_pass")
    def pass_risk_approval(self):
        print("分析风险，通过")
        return "pass_risk_approval"

    @listen(pass_risk_approval)
    def trade_management(self):
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
        self.state.trade_approval_flag = "同意买入" in result.raw
        return result.raw

    @router(trade_management)
    def trade_approval(self):
        if self.state.trade_approval_flag:
            return "trade_approval_pass"
        else:
            return "approval_reject"

    @listen("trade_approval_pass")
    def trade(self):
        print("允许交易")
        csv_file = "trade.csv"
        append_number_to_csv(csv_file, self.state.symbol)
        return "trade"

    @listen("approval_reject")
    def reject_approval(self):
        print("交易不通过")
        return "reject_approval"


def create_object(num):
    return {
        "symbol": num,
        "start_date": "20180101",
        "end_date": datetime.now().strftime("%Y%m%d"),
        "file_date": datetime.now().strftime("%Y%m%d%H%M%S"),
    }


def kickoff():
    numbers = [600050, 600690]
    symbols = list(map(create_object, numbers))
    for item in symbols:
        TradingState.symbol = item["symbol"]
        TradingState.start_date = item["start_date"]
        TradingState.end_date = item["end_date"]
        TradingState.file_date = item["file_date"]
        trading_flow = TradingFlow()
        trading_flow.kickoff()


def plot():
    trading_flow = TradingFlow()
    trading_flow.plot("trading_flow_plot")


if __name__ == "__main__":
    kickoff()
