import requests
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from typing import List, Dict

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re

import asyncio


def get_news(end_date):
    """
    使用浏览器渲染方式爬取东方财富财经频道页面数据。
    :param end_date: str, 格式 'YYYYMMDD'
    :return: str, 拼接的新闻字符串 (格式: 20251109 | 标题 | 概要)
    """

    # === Step 1: 计算目标日期 ===
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    weekday = end_dt.weekday()  # 0=Mon, 6=Sun
    if weekday == 6:  # 周日
        target_dates = [end_dt - timedelta(days=i) for i in range(3)]  # Fri/Sat/Sun
    elif weekday == 5:  # 周六
        target_dates = [end_dt - timedelta(days=i) for i in range(2)]  # Fri/Sat
    else:
        target_dates = [end_dt]
    target_strs = [d.strftime("%Y-%m-%d") for d in target_dates]

    # === Step 2: 浏览器渲染页面 ===
    all_items = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for page_num in range(1, 100, +1):  # 从第1页往后翻
            url = f"https://finance.eastmoney.com/a/ccjdd_{page_num}.html"
            page.goto(url, timeout=20000)
            # 等待页面加载出新闻列表
            page.wait_for_selector("ul#newsListContent li", timeout=10000)
            html = page.content()

            soup = BeautifulSoup(html, "html.parser")
            lis = soup.select("ul#newsListContent > li")

            for li in lis:
                title_el = li.select_one(".title a")
                info_el = li.select_one(".info")
                time_el = li.select_one(".time")

                if not (title_el and time_el):
                    continue

                title = title_el.get_text(strip=True)
                summary = info_el.get_text(strip=True) if info_el else ""
                time_text = time_el.get_text(strip=True)

                # 例: "11-09 15:34" -> 2025-11-09
                # pub_time = datetime.strptime(f"{time_text}", "%Y-%m-%d %H:%M")
                # pub_date_str = pub_time.strftime("%Y-%m-%d")

                # 用正则提取 “YYYY年MM月DD日 HH:MM”
                m = re.search(r"(\d{4})年(\d{2})月(\d{2})日\s*(\d{2}:\d{2})", time_text)
                if not m:
                    continue
                try:
                    dt = datetime.strptime(
                        f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}",
                        "%Y-%m-%d %H:%M",
                    )
                    pub_date_str = dt.strftime("%Y-%m-%d")
                except Exception as e:
                    continue
                if pub_date_str in target_strs:
                    all_items.append(f"{pub_date_str} | {title} | {summary}")

            # 判断是否可以提前结束
            last_time_el = lis[-1].select_one(".time")
            if last_time_el:
                time_text = last_time_el.get_text(strip=True)
                # 用正则提取 "YYYY年MM月DD日 HH:MM"
                m = re.search(r"(\d{4})年(\d{2})月(\d{2})日\s*(\d{2}:\d{2})", time_text)
                if m:
                    try:
                        last_time = datetime.strptime(
                            f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}",
                            "%Y-%m-%d %H:%M",
                        )
                        if last_time.date() < min(target_dates).date():
                            break
                    except Exception:
                        continue

        browser.close()

    return "\n".join(all_items)


def get_date_range_for_weekend(end_date: str):
    """
    根据传入的日期确定需要获取数据的日期范围

    Args:
        end_date: 日期字符串，格式为 "20251011"

    Returns:
        list: 需要获取数据的日期列表
    """
    try:
        # 将字符串转换为datetime对象
        date_obj = datetime.strptime(end_date, "%Y%m%d")

        # 获取星期几 (0=周一, 1=周二, ..., 5=周六, 6=周日)
        weekday = date_obj.weekday()

        # 根据星期几确定日期范围
        if weekday == 6:  # 周日
            # 获取周日、周六、周五
            dates = [
                date_obj.strftime("%Y%m%d"),
                (date_obj - timedelta(days=1)).strftime("%Y%m%d"),
                (date_obj - timedelta(days=2)).strftime("%Y%m%d"),
            ]
        elif weekday == 5:  # 周六
            # 获取周六、周五
            dates = [
                date_obj.strftime("%Y%m%d"),
                (date_obj - timedelta(days=1)).strftime("%Y%m%d"),
            ]
        else:
            # 工作日，只获取当天
            dates = [date_obj.strftime("%Y%m%d")]

        return dates

    except ValueError:
        # 日期格式错误，返回原日期
        return [end_date]


def get_mes_stock_events_analysis(symbol: str, end_date: str):
    """
    获取股票事件分析数据

    Args:
        symbol: 股票代码，如 "600745"
        end_date: 结束日期，格式为 "20251011"

    Returns:
        str: 拼接后的所有数据字符串
    """
    # 获取需要查询的日期范围
    date_range = get_date_range_for_weekend(end_date)

    # 构建请求参数
    params = {
        "args": {
            "market": "1",  # 默认沪市，可以根据需要调整
            "pageNumber": "1",
            "pageSize": "100",
            "securityCode": symbol,
            "fields": "code,infoCode,title,showDateTime,summary,uniqueUrl,url",
        },
        "method": "securityNews",
    }

    url = "https://emdcnewsapp.eastmoney.com/infoService"

    try:
        # 发送POST请求
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        response = requests.post(url, json=params, headers=headers, timeout=10)
        response.raise_for_status()

        # 解析响应数据
        data = response.json()

        if data.get("code") != 0:
            return f"接口返回错误: {data.get('code')}"

        items = data.get("data", {}).get("items", [])

        if not items:
            return "未找到相关数据"

        # 处理并过滤数据
        result_parts = []

        for item in items:
            # 转换时间格式
            show_time = item.get("showDateTime", 0)
            if show_time:
                # 将毫秒时间戳转换为日期
                dt = datetime.fromtimestamp(show_time / 1000)
                time_str = dt.strftime("%Y%m%d")
            else:
                time_str = "未知时间"

            # 只保留在date_range中的日期数据
            if time_str not in date_range:
                continue

            title = item.get("title", "无标题")
            summary = item.get("summary", "无摘要")

            # 拼接每条数据
            item_str = f"时间：{time_str}\n标题：{title}\n内容：{summary}\n{'-'*50}"
            result_parts.append(item_str)

        # 检查是否有符合条件的数据
        if not result_parts:
            date_range_str = "、".join(date_range)
            return f"未找到{date_range_str}的相关资讯数据"

        # 将所有数据拼接成一个字符串
        final_result = "\n".join(result_parts)
        return final_result

    except requests.exceptions.RequestException as e:
        return f"网络请求错误: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON解析错误: {str(e)}"
    except Exception as e:
        return f"处理数据时发生错误: {str(e)}"


def get_com_stock_events_analysis(symbol: str, end_date: str):
    """
    获取股票事件分析数据

    Args:
        symbol: 股票代码，如 "002326"
        end_date: 结束日期，格式为 "20251011"

    Returns:
        str: 拼接后的所有数据字符串
    """
    # 获取需要查询的日期范围
    date_range = get_date_range_for_weekend(end_date)

    # 根据股票代码判断市场类型
    if symbol.startswith(("0", "3")):
        # 深市股票
        market_prefix = "0"
    elif symbol.startswith(("6", "9")):
        # 沪市股票
        market_prefix = "1"
    else:
        return "不支持的股票代码格式"

    # 构建market_stock_list参数
    market_stock_list = f"{market_prefix}.{symbol}"

    # 构建请求URL
    url = f"https://np-anotice-pc.eastmoney.com/api/security/ann"

    params = {
        "page_index": "1",
        "page_size": "100",  # 获取更多数据
        "market_stock_list": market_stock_list,
        "client_source": "web",
        "business": "f10",
        "v": "05609776991531656",
    }

    try:
        # 发送GET请求
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        # 解析响应数据
        data = response.json()

        if "data" not in data or "list" not in data["data"]:
            return "未找到相关公告数据"

        items = data["data"]["list"]

        if not items:
            return "未找到相关公告数据"

        # 处理并过滤数据
        result_parts = []

        for item in items:
            # 转换时间格式
            sort_date = item.get("sort_date", "")
            if sort_date:
                try:
                    # 将 "2025-11-07 12:00:00" 转换为 "20251107"
                    dt = datetime.strptime(sort_date.split()[0], "%Y-%m-%d")
                    time_str = dt.strftime("%Y%m%d")
                except ValueError:
                    time_str = "未知时间"
            else:
                time_str = "未知时间"

            # 只保留在date_range中的日期数据
            if time_str not in date_range:
                continue

            title = item.get("title", "无标题")

            # 拼接每条数据
            item_str = f"时间：{time_str} 标题：{title}"
            result_parts.append(item_str)

        # 检查是否有符合条件的数据
        if not result_parts:
            date_range_str = "、".join(date_range)
            return f"未找到{date_range_str}的相关公告数据"

        # 将所有数据拼接成一个字符串
        final_result = "\n".join(result_parts)
        return final_result

    except requests.exceptions.RequestException as e:
        return f"网络请求错误: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON解析错误: {str(e)}"
    except Exception as e:
        return f"处理数据时发生错误: {str(e)}"


def get_stock_events(symbol: str, end_date: str):
    """
    获取股票事件分析数据

    Args:
        symbol: 股票代码，如 "600745"
        end_date: 结束日期，格式为 "20251011"

    Returns:
        str: 拼接后的所有数据字符串
    """
    # 调用两个函数获取数据
    com_events = get_com_stock_events_analysis(symbol, end_date)
    mes_events = get_mes_stock_events_analysis(symbol, end_date)
    news = get_news(end_date)
    # 获取日期范围用于显示
    date_range = get_date_range_for_weekend(end_date)
    date_range_str = "、".join(date_range)

    # 拼接两个结果
    final_result = f"查询日期范围: {date_range_str}\n\n公司公告:\n{com_events}\n\n相关资讯:\n{mes_events}\n\n财经资讯:\n{news}"
    return final_result
