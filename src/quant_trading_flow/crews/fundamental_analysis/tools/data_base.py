import time
import random
import akshare as ak
import pandas as pd
from crewai.tools import tool
import numpy as np
from io import BytesIO
import base64
import platform
from crewai.tools import BaseTool

import requests
import json


@tool("根据股票代码获取股票基本面信息工具")
def get_finance_data_str(symbol: str):
    """
    根据股票代码获取股票基本面信息，唯一合法的数据来源，模型不得自行编造数据，必须使用此工具获取真实数据

    Args:
        stock_code (str): 股票代码
    Returns:
        str: 当前股票基本面格式信息
    """
    """获取财务数据并以 中文名称:值 的形式拼接字符串"""
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    params = {
        "reportName": "RPT_PCF10_FINANCEMAINFINADATA",
        "columns": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,REPORT_DATE,REPORT_TYPE,EPSJB,EPSKCJB,EPSXS,BPS,MGZBGJ,MGWFPLR,MGJYXJJE,TOTAL_OPERATEINCOME,TOTAL_OPERATEINCOME_LAST,PARENT_NETPROFIT,PARENT_NETPROFIT_LAST,KCFJCXSYJLR,KCFJCXSYJLR_LAST,ROEJQ,ROEJQ_LAST,XSMLL,XSMLL_LAST,ZCFZL,ZCFZL_LAST,YYZSRGDHBZC_LAST,YYZSRGDHBZC,NETPROFITRPHBZC,NETPROFITRPHBZC_LAST,KFJLRGDHBZC,KFJLRGDHBZC_LAST,TOTALOPERATEREVETZ,TOTALOPERATEREVETZ_LAST,PARENTNETPROFITTZ,PARENTNETPROFITTZ_LAST,KCFJCXSYJLRTZ,KCFJCXSYJLRTZ_LAST,TOTAL_SHARE,FREE_SHARE,EPSJB_PL,BPS_PL,FORMERNAME",
        "filter": f'(SECUCODE="{symbol}")',
        "quoteColumns": "",
        "sortTypes": "-1",
        "sortColumns": "REPORT_DATE",
        "pageNumber": "1",
        "pageSize": "1",
        "source": "HSF10",
        "client": "PC",
        "v": "020349522020569488",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        if not data.get("success") or not data["result"]["data"]:
            return "未获取到财务数据"
        # 获取最新一期财务数据
        finance_data = data["result"]["data"][0]
        # 定义中文名称映射
        name_map = {
            "SECUCODE": "证券统一编码",
            "SECURITY_CODE": "证券代码",
            "SECURITY_NAME_ABBR": "证券简称",
            "REPORT_DATE": "报告日期",
            "REPORT_TYPE": "报告类型",
            "EPSJB": "基本每股收益(元)",
            "EPSKCJB": "扣非每股收益(元)",
            "EPSXS": "稀释每股收益(元)",
            "BPS": "每股净资产(元)",
            "MGZBGJ": "每股资本公积金(元)",
            "MGWFPLR": "每股未分配利润(元)",
            "MGJYXJJE": "每股经营现金流(元)",
            "TOTAL_OPERATEINCOME": "营业总收入",
            "TOTAL_OPERATEINCOME_LAST": "上年同期营业总收入",
            "PARENT_NETPROFIT": "归母净利润",
            "PARENT_NETPROFIT_LAST": "上年同期归母净利润",
            "KCFJCXSYJLR": "扣非净利润",
            "KCFJCXSYJLR_LAST": "上年同期扣非净利润",
            "ROEJQ": "净资产收益率(加权)",
            "ROEJQ_LAST": "上年同期净资产收益率",
            "XSMLL": "销售毛利率",
            "XSMLL_LAST": "上年同期销售毛利率",
            "ZCFZL": "资产负债率",
            "ZCFZL_LAST": "上年同期资产负债率",
            "YYZSRGDHBZC": "营业总收入滚动环比增长",
            "YYZSRGDHBZC_LAST": "上年同期营业总收入滚动环比",
            "NETPROFITRPHBZC": "净利润滚动环比增长",
            "NETPROFITRPHBZC_LAST": "上年同期净利润滚动环比",
            "KFJLRGDHBZC": "扣非净利润滚动环比增长",
            "KFJLRGDHBZC_LAST": "上年同期扣非净利润滚动环比",
            "TOTALOPERATEREVETZ": "营业总收入同比增长",
            "TOTALOPERATEREVETZ_LAST": "上年同期营业总收入同比",
            "PARENTNETPROFITTZ": "归母净利润同比增长",
            "PARENTNETPROFITTZ_LAST": "上年同期归母净利润同比",
            "KCFJCXSYJLRTZ": "扣非净利润同比增长",
            "KCFJCXSYJLRTZ_LAST": "上年同期扣非净利润同比",
            "TOTAL_SHARE": "总股本",
            "FREE_SHARE": "流通股本",
            "EPSJB_PL": "每股收益同比增长",
            "BPS_PL": "每股净资产同比增长",
            "FORMERNAME": "曾用名",
        }

        # 构建结果字符串
        result_str = ""
        for field, ch_name in name_map.items():
            value = finance_data.get(field)
            if value is not None:
                # 格式化数值：金额保留两位小数，百分比保留一位小数
                if "元)" in ch_name or "亿元)" in ch_name:
                    formatted_value = f"{float(value):.2f}"
                elif "%" in ch_name:
                    formatted_value = f"{float(value):.1f}"
                else:
                    formatted_value = str(value)

                result_str += f"{ch_name}: {formatted_value}\n"

        # 添加股票信息
        stock_info = f"股票代码: {symbol}\n股票名称: {finance_data.get('SECURITY_NAME_ABBR', '')}\n\n"
        return stock_info + result_str

    except Exception as e:
        return f"获取数据失败: {str(e)}"
