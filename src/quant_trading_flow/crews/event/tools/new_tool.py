import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re


def get_news(end_date: str):
    """
    爬取东方财富网财经新闻，获取指定日期及前两天数据（周末逻辑）
    参数：
        end_date (str): 日期字符串，例如 '2025-11-09'
    返回：
        拼接字符串：时间（20251011）|标题|概要
    """
    base_url = "https://finance.eastmoney.com/a/ccjdd_15.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(base_url, headers=headers)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    weekday = end_dt.weekday()  # 0=周一, 6=周日

    # 获取需要对比的日期集合
    dates_to_match = []
    if weekday == 6:  # 周日
        dates_to_match = [
            end_dt - timedelta(days=i) for i in range(3)
        ]  # 周日、周六、周五
    elif weekday == 5:  # 周六
        dates_to_match = [end_dt - timedelta(days=i) for i in range(2)]  # 周六、周五
    else:
        dates_to_match = [end_dt]

    date_strs = [d.strftime("%Y%m%d") for d in dates_to_match]

    result = []

    # 查询新闻列表（ul#newsListContent > li）
    for li in soup.select("#newsListContent li"):
        title_tag = li.select_one(".title a")
        info_tag = li.select_one(".info")
        time_tag = li.select_one(".time")

        if not (title_tag and info_tag and time_tag):
            continue

        raw_time = time_tag.get_text(strip=True)
        # 用正则提取 “YYYY年MM月DD日 HH:MM”
        m = re.search(r"(\d{4})年(\d{2})月(\d{2})日\s*(\d{2}:\d{2})", raw_time)
        if not m:
            continue

        try:
            dt = datetime.strptime(
                f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}", "%Y-%m-%d %H:%M"
            )
            date_str = dt.strftime("%Y%m%d")
        except Exception as e:
            continue

        if date_str in date_strs:
            title = title_tag.get_text(strip=True)
            info = info_tag.get_text(strip=True)
            result.append(f"{date_str} | {title} | {info}")

    return "\n".join(result)


# 示例：
if __name__ == "__main__":
    print(get_news("2025-11-09"))
