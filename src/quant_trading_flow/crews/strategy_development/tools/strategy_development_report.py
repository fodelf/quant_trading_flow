from crewai.tools import tool


@tool("读取本地文件获取交易数据报告工具")
def get_data_report(symbol: str, file_date: str) -> str:
    """
    读取本地文件获取交易数据报告

    Args:
        symbol (str): 股票代码
        file_date (str): 文件日期

    Returns:
        str: 返回交易数据报告报告字符串
    """
    with open(f"output/{symbol}/{file_date}/data_report.md", "r") as f:
        data_report = f.read()
    return data_report


# @tool("读取本地文件获取基本面数据报告")
# def get_data_analysis(symbol: str, file_date: str) -> str:
#     """
#     读取本地文件获取基本面数据报告

#     Args:
#       symbol (str): 股票代码
#       file_date (str): 文件日期

#     Returns:
#         str: 返回基本面数据报告报告字符串
#     """
#     with open(f"output/{symbol}/{file_date}/data_analysis.md", "r") as f:
#         data_analysis = f.read()
#     return data_analysis
