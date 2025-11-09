import requests
import json
import datetime
from typing import Optional


def get_event_stock_events_analysis(symbol: str, end_date: str) -> str:
    """
    获取股票事件分析数据，返回与指定日期最接近一天的所有数据

    Args:
        symbol: 股票代码，如 "600745"
        end_date: 目标日期，格式为 "20251011"

    Returns:
        格式化后的字符串，包含时间、标题和摘要信息
    """
    try:
        # 构造请求参数
        params = {
            "args": {
                "market": "1",
                "pageNumber": "1",
                "pageSize": "100",
                "securityCode": symbol,
                "fields": "code,infoCode,title,showDateTime,summary,uniqueUrl,url",
            },
            "method": "securityNews",
        }

        # 发送请求
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        response = requests.post(
            "https://emdcnewsapp.eastmoney.com/infoService",
            json=params,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()

        # 检查响应状态
        if data.get("code") != 0:
            return ""

        items = data.get("data", {}).get("items", [])
        if not items:
            return ""

        # 将目标日期转换为datetime对象
        target_date = datetime.datetime.strptime(end_date, "%Y%m%d")

        # 存储符合条件的日期数据
        candidate_dates = {}

        for item in items:
            show_timestamp = item.get("showDateTime")
            if not show_timestamp:
                continue

            # 将时间戳转换为datetime对象（毫秒转秒）
            show_time = datetime.datetime.fromtimestamp(show_timestamp / 1000)
            show_date_str = show_time.strftime("%Y%m%d")

            # 计算日期差
            date_diff = (target_date - show_time).days

            # 只考虑当天或前一天的数据
            if abs(date_diff) <= 1:
                if show_date_str not in candidate_dates:
                    candidate_dates[show_date_str] = []
                candidate_dates[show_date_str].append(item)

        # 确定最终选择的日期
        selected_date = None
        if end_date in candidate_dates:  # 优先选择当天
            selected_date = end_date
        else:
            # 查找前一天
            previous_date = (target_date - datetime.timedelta(days=1)).strftime(
                "%Y%m%d"
            )
            if previous_date in candidate_dates:
                selected_date = previous_date

        if not selected_date or selected_date not in candidate_dates:
            return ""

        # 构建结果字符串
        result_parts = []
        for item in candidate_dates[selected_date]:
            title = item.get("title", "")
            summary = item.get("summary", "")

            # 格式化每个条目
            item_str = f"{selected_date}{title}（{summary}）"
            result_parts.append(item_str)

        return "".join(result_parts)

    except requests.exceptions.RequestException:
        return ""
    except json.JSONDecodeError:
        return ""
    except ValueError:
        return ""
    except Exception:
        return ""


# # 使用示例
# if __name__ == "__main__":
#     # 测试函数
#     result = get_event_stock_events_analysis("600745", "20251011")
#     print("结果:", result)

#     # 如果没有数据的情况
#     empty_result = get_event_stock_events_analysis("000001", "20251011")
#     print("空结果:", empty_result)
