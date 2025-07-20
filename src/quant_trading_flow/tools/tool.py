from datetime import datetime
import os
import csv
import re
import json
import os
from typing import Union, List, Dict, Optional
import json5
import pandas as pd


def delete_csv_by_value(csv_path, Value):
    df = pd.read_csv(csv_path)
    print(df)
    # 过滤掉目标值所在的行
    filtered_df = df[df["Value"] != int(Value)]
    print(filtered_df)
    # 保存结果到新文件
    filtered_df.to_csv(csv_path, index=False)


def read_csv_values(csv_path: str, column_name: str = "Value") -> List[str]:
    """
    从CSV文件中读取指定列的值

    参数:
        csv_path: CSV文件路径
        column_name: 要读取的列名 (默认为"Value")

    返回:
        去重后的值列表
    """
    values = set()

    if not os.path.exists(csv_path):
        print(f"错误: 文件 '{csv_path}' 不存在")
        return list(values)

    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if column_name in row:
                    value = row[column_name].strip()
                    if value:  # 确保值不为空
                        values.add(value)
        return list(values)
    except Exception as e:
        print(f"读取CSV时出错: {e}")
        return []


def create_object(num, current_price):
    return {
        "current_price": current_price,
        "has_flag": True,
        "trade_flag": False,
        "symbol": num,
        "start_date": "20180101",
        "end_date": datetime.now().strftime("%Y%m%d"),
        # "file_date": "20250713220455",
        "file_date": datetime.now().strftime("%Y%m%d%H%M%S"),
    }


def create_object_default(num):
    return {
        "has_flag": False,
        "trade_flag": False,
        "current_price": 0,
        "symbol": num,
        "start_date": "20180101",
        "end_date": datetime.now().strftime("%Y%m%d"),
        # "file_date": "20250713220455",
        "file_date": datetime.now().strftime("%Y%m%d%H%M%S"),
    }


def create_object_trade(num):
    return {
        "has_flag": False,
        "trade_flag": True,
        "current_price": 0,
        "symbol": num,
        "start_date": "20180101",
        "end_date": datetime.now().strftime("%Y%m%d"),
        "file_date": datetime.now().strftime("%Y%m%d%H%M%S"),
    }


def extract_json_from_text1(text):
    """使用JSON5解析器处理非标准JSON"""
    json_match = re.search(r"\{[\s\S]*?\}", text)
    if not json_match:
        return None

    try:
        return json5.loads(json_match.group(0))
    except Exception as e:
        print("JSON5解析失败:", str(e))
        return None


def extract_json_from_text(text):
    """
    从混合文本中提取并解析JSON数据
    :param text: 包含JSON和其他文本的字符串
    :return: 解析后的Python对象, 原始JSON字符串
    """
    # 方法1：使用正则表达式提取JSON部分
    json_match = re.search(r"\{[\s\S]*?\}", text)

    if not json_match:
        # 如果找不到完整JSON对象，尝试提取数组部分
        array_match = re.search(r"\[[\s\S]*?\]", text)
        if array_match:
            # 将数组包装成完整JSON对象
            json_str = f'{{"stock_list": {array_match.group(0)}}}'
            return json.loads(json_str), json_str
        else:
            raise ValueError("未找到有效的JSON数据")

    json_str = json_match.group(0)

    try:
        # 尝试直接解析JSON
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
        print("尝试修复JSON...")

        # 修复常见JSON问题
        fixed_json = json_str

        # 1. 确保属性名用双引号包裹
        fixed_json = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'"\1":', fixed_json)

        # 2. 将单引号替换为双引号
        fixed_json = fixed_json.replace("'", '"')

        # 3. 修复尾随逗号问题
        fixed_json = re.sub(r",\s*([}\]])", r"\1", fixed_json)

        # 4. 移除可能存在的注释
        fixed_json = re.sub(r"//.*?$", "", fixed_json, flags=re.MULTILINE)
        fixed_json = re.sub(r"/\*.*?\*/", "", fixed_json, flags=re.DOTALL)

        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError as e2:
            print(f"修复后仍然失败: {str(e2)}")
            print("修复后的JSON字符串:")
            print(fixed_json)
            raise


def extract_stock_info(text):
    """
    提取股票代码和相关信息
    :param text: 包含股票信息的文本
    :return: 字典 {股票代码: 信息}
    """
    stock_info = {}
    # 查找模式: 6位数字 + 中文冒号 + 任意文本
    pattern = r"(\d{6})：(.+?)(?=\d{6}：|$)"

    for match in re.finditer(pattern, text):
        stock_code = match.group(1)
        info = match.group(2).strip()
        stock_info[stock_code] = info

    return stock_info


def append_number_to_csv(file_path, number):
    """
    向CSV文件追加数字：如果文件不存在则创建并写入，如果存在则追加

    参数:
        file_path (str): CSV文件路径
        number (int/float): 要追加的数字
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 文件不存在，创建新文件并写入数字
        df = pd.DataFrame([number], columns=["Value"])
        df.to_csv(file_path, index=False)
        print(f"✅ 创建新文件 '{file_path}' 并写入初始值: {number}")
    else:
        try:
            # 文件存在，读取现有数据
            df = pd.read_csv(file_path)

            # 创建新行并追加
            new_row = pd.DataFrame([number], columns=["Value"])
            df = pd.concat([df, new_row], ignore_index=True)

            # 保存回文件
            df.to_csv(file_path, index=False)
            print(f"📝 在 '{file_path}' 中追加值: {number}")

        except Exception as e:
            print(f"❌ 追加数据失败: {e}")


def merge_stock_lists(
    csv_values: List[str], json_stocks: List[str], format_codes: bool = True
) -> List[str]:
    """
    合并并去重两个股票列表

    参数:
        csv_values: 从CSV读取的值列表
        json_stocks: JSON中的股票列表
        format_codes: 是否统一格式化股票代码

    返回:
        合并去重后的排序股票列表
    """
    # 创建集合用于去重
    unique_stocks = set()

    # 添加CSV值
    for stock in csv_values:
        if format_codes:
            # 统一格式化为6位数字符串，不足补零
            stock = stock.zfill(6)
        unique_stocks.add(stock)

    # 添加JSON股票
    for stock in json_stocks:
        if format_codes:
            stock = stock.zfill(6)
        unique_stocks.add(stock)

    # 转换为列表并排序
    sorted_stocks = sorted(unique_stocks)
    return sorted_stocks
