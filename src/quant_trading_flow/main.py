#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, and_, router

from datetime import datetime

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
from quant_trading_flow.crews.risk_management.risk_has_management import (
    RiskHasManagementCrew,
)

from quant_trading_flow.crews.cfo.cfo import CfoCrew
from quant_trading_flow.crews.cfo.cfo_has import CfoHasCrew

from quant_trading_flow.crews.stock_screener.stock_screener import StockScreenerCrew
from quant_trading_flow.tools.tool import (
    read_csv_values,
    create_object,
    create_object_default,
    create_object_trade,
    extract_json_from_text,
    append_number_to_csv,
    delete_csv_by_value,
)


class TradingState(BaseModel):
    symbol: str = "SHA:600513"
    start_date: str = "20180101"
    end_date: str = datetime.now().strftime("%Y%m%d")
    file_date: str = datetime.now().strftime("%Y%m%d%H%M%S")
    risk_approval_flag: bool = True
    trade_approval_flag: bool = True
    has_flag: bool = False
    current_price: float = 0.0
    trade_flag: bool = False


def getData(state):
    return {
        "symbol": state.symbol,
        "start_date": state.start_date,
        "end_date": state.end_date,
        "file_date": state.file_date,
        "current_price": state.current_price,
        "has_flag": state.has_flag,
        "trade_flag": state.trade_flag,
    }


class TradingFlow(Flow[TradingState]):

    # @start()
    # def init_market_data1(self):
    #     return getData(self.state)

    @start()
    def init_market_data(self):
        return getData(self.state)

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
        result = StrategyDevelopmentCrew().crew().kickoff(inputs=getData(self.state))
        return result.raw

    @router(
        and_(
            strategy_development,
            public_sentiment,
            fundamental_analysis,
            government_affairs,
        )
    )
    def risk_management_decision(self):
        if self.state.has_flag:
            return "risk_has_management_decision"
        else:
            return "risk_default_management_decision"

    @listen("risk_default_management_decision")
    def risk_management(self):
        result = RiskManagementCrew().crew().kickoff(inputs=getData(self.state))
        self.state.risk_approval_flag = "同意买入" in result.raw
        return result.raw

    @listen("risk_has_management_decision")
    def risk_has_management(self):
        result = RiskHasManagementCrew().crew().kickoff(inputs=getData(self.state))
        return result.raw

    @router(risk_management)
    def risk_approval(self):
        if self.state.risk_approval_flag:
            return "risk_approval_pass"
        else:
            return "approval_reject"

    @router("risk_approval_pass")
    def trade_management(self):
        result = CfoCrew().crew().kickoff(inputs=getData(self.state))
        self.state.trade_approval_flag = "同意买入" in result.raw
        if self.state.trade_approval_flag:
            return "trade_approval_pass"
        else:
            return "approval_reject"

    @listen(risk_has_management)
    def trade_has_management(self):
        result = CfoHasCrew().crew().kickoff(inputs=getData(self.state))
        self.state.trade_approval_flag = "同意卖出" in result.raw
        return result.raw

    @router("trade_approval_pass")
    def trade_decision(self):
        if self.state.trade_flag:
            return "wait_trade_pass"
        else:
            return "join_trade_pass"

    @listen("join_trade_pass")
    def join_trade(self):
        print("加入备选准备交易")
        csv_file = "trade.csv"
        append_number_to_csv(csv_file, self.state.symbol)

    @listen("wait_trade_pass")
    def waite_trade(self):
        print("维持现状等待交易")

    @router("approval_reject")
    def reject_decision(self):
        if self.state.trade_flag:
            return "has_reject_approval"
        else:
            return "no_reject_approval"

    @listen("no_reject_approval")
    def reject_trade(self):
        print("不允许参加交易")
        csv_file = "no_trade.csv"
        append_number_to_csv(csv_file, self.state.symbol)

    @listen("has_reject_approval")
    def delete_reject(self):
        print("踢出交易备选")
        csv_file = "trade.csv"
        delete_csv_by_value(csv_file, self.state.symbol)

    @router(trade_has_management)
    def trade_has_approval(self):
        if self.state.trade_approval_flag:
            return "trade_sell"
        else:
            return "trade_handel"

    @listen("trade_sell")
    def sell_immediate(self):
        print("立即卖出")
        return "sell"

    @listen("trade_handel")
    def handel_time(self):
        print("持续持有")
        return "handel"


def runTask(list):
    for item in list:
        TradingState.symbol = item["symbol"]
        TradingState.start_date = item["start_date"]
        TradingState.end_date = item["end_date"]
        TradingState.file_date = item["file_date"]
        TradingState.current_price = item["current_price"]
        TradingState.has_flag = item["has_flag"]
        TradingState.trade_flag = item["trade_flag"]
        trading_flow = TradingFlow()
        trading_flow.kickoff()


def kickoff():
    # 已经持有
    # csv_values = read_csv_values("has_trade.csv", "Value")
    # print(csv_values)
    # handle_values = read_csv_values("has_trade.csv", "Handle")
    # print(handle_values)
    csv_values = ["600690"]
    handle_values = ["24.85"]
    csv_values_symbols = list(map(create_object, csv_values, handle_values))
    print(csv_values_symbols)
    runTask(csv_values_symbols)
    # return
    # return
    # 尚未投资
    csv_values = read_csv_values("trade.csv")
    csv_values_symbols = list(map(create_object_trade, csv_values))
    runTask(csv_values_symbols)
    result = (
        StockScreenerCrew()
        .crew()
        .kickoff(inputs={"end_date": datetime.now().strftime("%Y%m%d")})
    )
    extract_json = extract_json_from_text(result.raw)
    stock_list = extract_json.get("stock_list", [])
    symbols = list(map(create_object_default, stock_list))
    runTask(symbols)


def plot():
    trading_flow = TradingFlow()
    trading_flow.plot("trading_flow_plot")


if __name__ == "__main__":
    kickoff()
