import subprocess
import time

# 指令列表
commands = [
    # # 跑MI被试内
    # "python main.py --exp_setting within_sub --dataset_name 2a",
    # "python main.py --exp_setting within_sub --dataset_name 2b",
    # "python main.py --exp_setting within_sub --dataset_name BNCI2014_002",
    # "python main.py --exp_setting within_sub --dataset_name BNCI2015_001",
    # "python main.py --exp_setting within_sub --dataset_name BNCI2015_001",
    "python main.py --exp_setting within_sub --dataset_name Weibo2014",

    # # 跑2a数据集尝试不同示例数量
    # "python main.py --exp_setting within_sub --dataset_name 2a --num_demos 2",
    # "python main.py --exp_setting within_sub --dataset_name 2a --num_demos 8",
    # "python main.py --exp_setting within_sub --dataset_name 2a --num_demos 16",

    # 跑BNCI2015001数据集使用deepseek模型
    # "python main.py --exp_setting within_sub --dataset_name BNCI2015_001 --model_type deepseek-chat --methods BL --repeat_times 5",

    # # 跑MI跨被试
    # "python main.py --exp_setting cross_sub --dataset_name 2a",
    # "python main.py --exp_setting cross_sub --dataset_name 2b",
    # "python main.py --exp_setting cross_sub --dataset_name BNCI2014_002",
    # "python main.py --exp_setting cross_sub --dataset_name BNCI2015_001",
    # "python main.py --exp_setting cross_sub --dataset_name Weibo2014",

    # sleep被试内与跨被试
    # "python main.py --exp_setting within_sub --dataset_name sleep-edfx --num_demos 10", # 5分类问题
    # "python main.py --exp_setting cross_sub --dataset_name sleep-edfx --num_demos 10", # 5分类问题

    # chb-mit 癫痫数据集
    # "python main.py --exp_setting within_sub --dataset_name chb-mit --num_demos 4", # 2分类问题
]

# 重试次数设置
max_retries = 3

# 依次执行每条指令
for cmd in commands:
    print(f"Running: {cmd}")
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            # 注意：check=True，表示如果命令返回非0（错误），会抛异常
            subprocess.run(cmd, shell=True, check=True)
            success = True
            print(f"Success on attempt {attempt}: {cmd}")
            break  # 成功就跳出 retry 循环
        except subprocess.CalledProcessError:
            print(f"Attempt {attempt} failed for: {cmd}")
            if attempt < max_retries:
                time.sleep(5)  # 重试前等待几秒（可选）
            else:
                print(f"All {max_retries} attempts failed for: {cmd}")
    print()

print("All commands finished.")