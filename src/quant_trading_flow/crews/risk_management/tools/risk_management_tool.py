from crewai.tools import tool


@tool("读取本地文件获取交易数据报告")
def get_data_report(symbol: str, file_date: str) -> str:
    """
    读取本地文件获取交易数据报告

    Args:
        symbol (str): 股票代码
        file_date (str): 文件日期

    Returns:
        str: 返回交易数据报告字符串
    """
    with open(f"output/{symbol}/{file_date}/data_report.md", "r") as f:
        data_report = f.read()
    return data_report


@tool("读取本地文件获取基本面数据报告")
def get_data_analysis(symbol: str, file_date: str) -> str:
    """
    读取本地文件获取基本面数据报告

    Args:
        symbol (str): 股票代码
        file_date (str): 文件日期

    Returns:
        str: 返回基本面数据报告字符串
    """
    with open(f"output/{symbol}/{file_date}/data_analysis.md", "r") as f:
        data_analysis = f.read()
    return data_analysis


@tool("读取本地文件获取政府政策与市场环境报告")
def get_government_affairs(symbol: str, file_date: str) -> str:
    """
    读取本地文件获取政府政策与市场环境报告

    Args:
        symbol (str): 股票代码
        file_date (str): 文件日期

    Returns:
        str: 返回政府政策与市场环境报告字符串
    """
    with open(f"output/{symbol}/{file_date}/government_affairs.md", "r") as f:
        government_affairs = f.read()
    return government_affairs


@tool("读取本地文件获取市场舆情报告")
def get_public_sentiment(symbol: str, file_date: str) -> str:
    """
    读取本地文件获取市场舆情报告

    Args:
        symbol (str): 股票代码
        file_date (str): 文件日期

    Returns:
        str: 返回市场舆情报告字符串
    """
    with open(f"output/{symbol}/{file_date}/public_sentiment.md", "r") as f:
        public_sentiment = f.read()
    return public_sentiment


@tool("读取本地文件获取策略数据报告")
def get_strategy_report(symbol: str, file_date: str) -> str:
    """
    读取本地文件获取策略数据报告

    Args:
        symbol (str): 股票代码
        file_date (str): 文件日期

    Returns:
        str: 返回策略数据报告字符串
    """
    with open(f"output/{symbol}/{file_date}/strategy_report.md", "r") as f:
        strategy_report = f.read()
    return strategy_report
