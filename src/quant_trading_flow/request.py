import requests
import json
import time
import hashlib


def generate_fingerprint():
    """生成动态 fingerprint 参数（示例逻辑，实际需要分析JS生成规则）"""
    timestamp = str(int(time.time() * 1000))
    raw_str = f"eastmoney_{timestamp}_smarttag"
    return hashlib.md5(raw_str.encode()).hexdigest()


def get_search_results(keyword="上证150", page_size=50, page_no=1):
    url = "https://np-tjxg-g.eastmoney.com/api/smart-tag/stock/v3/pw/search-code"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Origin": "https://data.eastmoney.com",
        "Referer": "https://data.eastmoney.com/",
        "Content-Type": "application/json",
    }

    timestamp = str(int(time.time() * 1000000))  # 微秒级时间戳
    request_id = (
        f"req_{hashlib.md5(timestamp.encode()).hexdigest()[:20]}_{timestamp[:13]}"
    )

    payload = {
        "keyWord": keyword,
        "pageSize": page_size,
        "pageNo": page_no,
        "fingerprint": generate_fingerprint(),
        "gids": [],
        "matchWord": "",
        "timestamp": timestamp,
        "shareToGuba": False,
        "requestId": request_id,
        "needCorrect": True,
        "removedConditionIdList": [],
        "xcId": f"xc{hashlib.md5(timestamp.encode()).hexdigest()[:10]}",
        "ownSelectAll": False,
        "dxInfo": [],
        "extraCondition": "",
        "sortName": "NEWEST_PRICE",
        "sortWay": "desc",
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=10
        )
        response.raise_for_status()

        # 解析结果
        result = response.json()
        if result.get("code") == 0:
            return result["data"]
        else:
            print(f"API返回错误: {result.get('message')}")
            return None

    except Exception as e:
        print(f"请求失败: {str(e)}")
        return None


# 使用示例
# if __name__ == "__main__":
#     data = get_search_results()
#     if data:
#         print(f"共找到 {data['total']} 条结果")
#         for item in data["list"]:
#             print(f"{item['code']}: {item['name']} - 最新价: {item['newestPrice']}")
