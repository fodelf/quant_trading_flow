import requests
import pandas as pd
import json
import time
import numpy as np
from tqdm import tqdm
from crewai.tools import tool


@tool("获取股票代码清单")
def get_filtered_stocks():
    """
    获取股票代码清单

    Args:

    Returns:
        str: 股票代码数组拼接的字符串
    """
    """
    获取符合以下筛选条件的股票数据：
    1. 总市值 >= 50亿
    2. 当前股价 <= 30元
    3. 人气排名 1-500
    """
    # 基础URL
    base_url = "https://data.eastmoney.com/dataapi/xuangu/list"

    # 请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://data.eastmoney.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Origin": "https://data.eastmoney.com",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    # 查询参数
    params = {
        "st": "CHANGE_RATE",  # 排序字段：涨跌幅
        "sr": "-1",  # 排序方向：降序
        "ps": "50",  # 每页大小
        "p": "1",  # 页码（初始为1）
        "sty": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,NEW_PRICE,CHANGE_RATE,VOLUME_RATIO,HIGH_PRICE,LOW_PRICE,PRE_CLOSE_PRICE,VOLUME,DEAL_AMOUNT,TURNOVERRATE,TOTAL_MARKET_CAP,NEW_PRICE,POPULARITY_RANK,DEBT_ASSET_RATIO,TOI_YOY_RATIO",  # 返回字段
        "filter": "(TOTAL_MARKET_CAP>=5000000000)(NEW_PRICE<=30.00)(POPULARITY_RANK>0)(POPULARITY_RANK<=500)",
        "source": "SELECT_SECURITIES",
        "client": "WEB",
    }

    # 存储所有股票数据
    all_stocks = []

    try:
        # 先获取第一页数据，确定总页数
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 检查是否有数据
        if "result" not in data or "data" not in data["result"]:
            print("未获取到有效数据")
            return pd.DataFrame()
        # 获取总记录数和总页数
        total_records = data["result"]["count"]
        page_size = params["ps"]
        total_pages = (total_records + int(page_size) - 1) // int(page_size)

        print(f"共找到 {total_records} 条记录，分 {total_pages} 页")

        # 添加第一页数据
        all_stocks.extend(data["result"]["data"])

        # 如果有多页，获取后续页面的数据
        if total_pages > 1:
            for page in tqdm(range(2, total_pages + 1), desc="获取分页数据"):
                # 更新页码
                params["p"] = str(page)

                # 获取当前页数据
                response = requests.get(
                    base_url, params=params, headers=headers, timeout=10
                )
                response.raise_for_status()
                page_data = response.json()

                # 添加当前页数据
                all_stocks.extend(page_data["result"]["data"])

                # 避免请求过快
                time.sleep(0.5)

        # 转换为DataFrame
        df = pd.DataFrame(all_stocks)

        # 重命名列为中文
        column_mapping = {
            "SECUCODE": "证券代码",
            "SECURITY_CODE": "股票代码",
            "SECURITY_NAME_ABBR": "股票名称",
            "NEW_PRICE": "最新价",
            "CHANGE_RATE": "涨跌幅",
            "VOLUME_RATIO": "量比",
            "HIGH_PRICE": "最高价",
            "LOW_PRICE": "最低价",
            "PRE_CLOSE_PRICE": "昨收",
            "VOLUME": "成交量",
            "DEAL_AMOUNT": "成交额",
            "TURNOVERRATE": "换手率",
            "TOTAL_MARKET_CAP": "总市值",
            "POPULARITY_RANK": "人气排名",
            "DEBT_ASSET_RATIO": "资产负债率",
            "TOI_YOY_RATIO": "营收同比增长",
        }
        df = df.rename(columns=column_mapping)

        # 转换数值列
        numeric_cols = [
            "最新价",
            "涨跌幅",
            "量比",
            "最高价",
            "最低价",
            "昨收",
            "成交量",
            "成交额",
            "换手率",
            "总市值",
            "人气排名",
            "资产负债率",
            "营收同比增长",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 转换市值单位为亿元
        df["总市值(亿)"] = df["总市值"] / 100000000
        df.drop(columns=["总市值"], inplace=True)

        # 添加日期列
        df["数据日期"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        return ",".join(df["股票代码"])

    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"发生未知错误: {e}")
        return pd.DataFrame()


# if __name__ == "__main__":
#     print("开始获取符合筛选条件的股票数据...")
#     stock_df = get_filtered_stocks()

#     if stock_df is not None and not stock_df.empty:
#         # 显示前10条数据
#         print("\n前10条筛选结果:")
#         print(stock_df.head(10))

#         # 保存到Excel
#         excel_file = save_to_excel(stock_df)

#         # 分析结果
#         analyze_results(stock_df)
#     else:
#         print("未获取到符合条件的数据")
