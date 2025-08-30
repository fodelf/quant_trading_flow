import requests
import re
import json
import time
from datetime import datetime, timedelta


def get_stock_events_analysis(stock_code, target_date_str):
    # 生成时间戳和回调函数名
    timestamp = str(int(time.time() * 1000))
    callback_name = f"jQuery3510{timestamp}_{timestamp[:-3]}"

    # 构造请求参数
    params = {
        "cb": callback_name,
        "param": json.dumps(
            {
                "uid": "",
                "keyword": stock_code,
                "type": ["cmsArticleWebOld"],
                "client": "web",
                "clientType": "web",
                "clientVersion": "curr",
                "param": {
                    "cmsArticleWebOld": {
                        "searchScope": "default",
                        "sort": "default",
                        "pageIndex": 1,
                        "pageSize": 10,  # 增加页面大小以获取更多结果
                        "preTag": "<em>",
                        "postTag": "</em>",
                    }
                },
            }
        ),
        "_": timestamp,
    }

    # 发送请求
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.eastmoney.com/",
    }
    res = ""
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        # 使用正则表达式提取JSON数据
        jsonp_pattern = re.compile(rf"^{callback_name}\((.*)\);?$")
        match = jsonp_pattern.match(response.text)
        if match:
            json_data = json.loads(match.group(1))

            # 提取并过滤数据
            if "result" in json_data and "cmsArticleWebOld" in json_data["result"]:
                articles = json_data["result"]["cmsArticleWebOld"]

                # 将输入日期转换为datetime对象
                input_date = datetime.strptime(target_date_str, "%Y%m%d")

                # 确定要过滤的日期范围
                target_dates = []

                # 检查是否为周末（周六或周日）
                # if input_date.weekday() in [5, 6]:  # 5=周六, 6=周日
                #     # 添加周五
                #     friday = input_date - timedelta(days=(input_date.weekday() - 4))
                #     target_dates.append(friday.strftime("%Y-%m-%d"))

                #     # 添加周末日期
                #     if input_date.weekday() == 5:  # 周六
                #         target_dates.append(input_date.strftime("%Y-%m-%d"))
                #     else:  # 周日
                #         saturday = input_date - timedelta(days=1)
                #         target_dates.append(saturday.strftime("%Y-%m-%d"))
                #         target_dates.append(input_date.strftime("%Y-%m-%d"))
                # else:
                #     # 非周末只添加当天
                #     target_dates.append(input_date.strftime("%Y-%m-%d"))

                # 过滤文章
                filtered_articles = [
                    article
                    for article in articles
                    # if any(article["date"].startswith(d) for d in target_dates)
                ]

                print(filtered_articles)
                # 格式化日期显示
                # date_range = "、".join([d.replace("-", "") for d in target_dates])
                for i, article in enumerate(filtered_articles, 1):
                    # 移除标题和内容中的高亮标签
                    clean_title = (
                        article["title"].replace("<em>", "").replace("</em>", "")
                    )
                    clean_content = (
                        article["content"].replace("<em>", "").replace("</em>", "")
                    )

                    # 提取纯日期部分 (去掉时间)
                    article_date = article["date"][:10].replace("-", "")

                    res += f"\n文章 {i} ({article_date}):\n"
                    res += f"标题: {clean_title}\n"
                    res += f"媒体: {article['mediaName']}\n"
                    res += f"时间: {article['date']}\n"
                    res += f"内容: {clean_content} \n"  # 只显示前100个字符
                    res += f"链接: {article['url']}\n"
                return res
            return "未检测到事件因子"

        else:
            print("Failed to parse JSONP response")
            return "未检测到事件因子"
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "未检测到事件因子"
