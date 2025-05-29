import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore
from collections import Counter
from active_prompt.prompt.llm import ask_llm_online, ask_llm_offline
from active_prompt.prompt.prompt_writer import prompt
import datetime

def extract_array_from_string(input_string):
    """
    从字符串中提取合法的 JSON 数组内容。

    参数:
    - data: str，包含 JSON 数据的字符串

    返回:
    - list，提取出的数组内容
    """
    try:
        # 使用正则表达式提取数组部分
        match = re.search(r'\[.*?\]', input_string, re.DOTALL)
        if match:
            json_array = match.group(0)  # 提取匹配的数组部分
            print("Extracted JSON:", json_array)  # 查看提取到的 JSON 字符串
            return json.loads(json_array)  # 将数组部分解析为 Python 列表
        else:
            raise ValueError("未找到合法的 JSON 数组")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解码失败: {e}")

def get_accuracy_and_log(y_true, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    # 计算精确度
    precision = precision_score(y_true, y_pred, average='macro')  # 'macro' 表示未加权的均值
    print(f"Precision: {precision}")

    # 计算召回率
    recall = recall_score(y_true, y_pred, average='macro')
    print(f"Recall: {recall}")

    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score: {f1}")

    return accuracy, precision, recall, f1

def most_common_element(lst):
    """
    返回列表中出现次数最多的元素
    :param lst: 输入列表
    :return: 出现次数最多的元素及其次数
    """
    if not lst:
        return None, 0  # 处理空列表情况

    count = Counter(lst)
    most_common = count.most_common(1)[0]  # 获取出现次数最多的元素
    return most_common[0], most_common[1]

def collect_y_pred(demo_data, demo_labels, predict_data, max_predict_num, model_type, is_model_online, label_names, mean_value, para):
    """
    将多次返回的lst汇总为一个lst
    demo_data: 用来做演示示例的4个trial,lst of ndArray
    demo_labels: 4个trial的label, lst
    predict_data: 待遇测的trial lst
    max_predict_num: 单次最多预测条数
    """
    y_pred = []
    ask_llm = ask_llm_online if is_model_online else ask_llm_offline
    for i in range (0, len(predict_data), max_predict_num):
        test_data = predict_data[i:i+max_predict_num] if i + max_predict_num <= len(predict_data) else predict_data[i:len(predict_data)]
        prompt(demo_data, demo_labels, test_data, label_names, mean_value, para)
        answer = ask_llm(model_type)
        print(f'第{(i // max_predict_num) + 1}段,API输出为:{answer}')
        y_pred_sub = extract_array_from_string(answer)
        if y_pred_sub:  # 确保提取的结果不为空
            y_pred.extend(y_pred_sub)
    return y_pred

def collect_y_pred_single(demo_data, demo_labels, predict_data, model_type, is_model_online, label_names):
    """
    将多次返回的lst汇总为一个lst
    demo_data: 用来做演示示例的4个trial,lst of ndArray
    demo_labels: 4个trial的label, lst
    predict_data: 待遇测的trial lst
    max_predict_num: 单次最多预测条数
    """
    ask_llm = ask_llm_online if is_model_online else ask_llm_offline
    prompt(demo_data, demo_labels, predict_data, label_names)
    answer = ask_llm(model_type)
    y_pred_sub = extract_array_from_string(answer)
    if y_pred_sub:  # 确保提取的结果不为空
        return y_pred_sub