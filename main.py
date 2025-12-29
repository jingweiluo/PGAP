import numpy as np # type: ignore
import random
import sys
import os
import time
import argparse
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_parent_dir)
from PGAP.prompt.prompt_writer import prompt
from utils import collect_y_pred, get_accuracy_and_log
from PGAP.load.load_data import get_seed_data
from PGAP.load.load_moabb import get_moabb_data, get_moabb_data_crs_sub
from PGAP.load.load_five_folds import get_moabb_data_cv
from PGAP.load.load_sleep import get_sleep_data, get_sleep_data_crs_sub
from PGAP.load.load_epilepsy import get_ep_data
from PGAP.load.load_uci import get_uci_data_cv
from PGAP.query.qbc import find_centroids, rd_basic
from PGAP.query.svm import balanced_svm, svm_near, bal_simn_multi
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--exp_setting', type=str, default='within_sub', help='')
    parser.add_argument('--dataset_name', type=str, default='2a', help='')
    parser.add_argument('--model_type', type=str, default='qwen2.5-14b-instruct-1m', help='')
    parser.add_argument('--num_demos', type=int, default=4, help='Total number of ICL demonstrations')
    parser.add_argument('--sub_index', type=int, default=None, help='subject id')
    parser.add_argument('--repeat_times', type=int, default=5, help='')
    # parser.add_argument('--methods', type=list, default=['BL', 'RD', 'CTD', 'SVM-far', 'SVM-near', 'PURE'], help='')
    parser.add_argument('--methods', type=str, nargs='+', default=['BL', 'RD', 'CTD', 'SVM-far', 'SVM-near', 'PURE'], help='')

    parser.add_argument('--max_predict_num', type=int, default=10, help='number of test samples to predict in one prompt')
    parser.add_argument('--test_id', type=int, default=1, help='for cross-session setting, 1 means use default test set as test, vise versa')
    parser.add_argument('--is_model_online', type=bool, default=True, help='use online model')
    parser.add_argument('--measurement', type=str, default='ed', help='ed for Euclidean, cs for Cosine similarity')
    parser.add_argument('--combine_method', type=str, default=None, help='')
    parser.add_argument('--test_num', type=int, default=None, help='only for small test')
    parser.add_argument("--noise_target", type=int, nargs="+", default=[0, 1])
    parser.add_argument('--ablation', type=str, default=None, help='feat_mask, feat_noise, feat_perm, filp_half_labels, filp_all_labels')

    return parser.parse_args()

def ndToList(array):
    '''
    将d维度ndArray转换成包含(d-1)维ndArray的list
    '''
    return [array[i] for i in range(array.shape[0])]

def transformIndices(selected_idx_dict, combine_method):
    '''
    selected_idx_dict: {
        category1: [0,3,4,5]
        category2: [1,8,10,12]
        ...
    }
    目前就是单纯把dict中的内容整合到一个list
    后面可以跟据combine_method调整index的顺序
    '''
    result = []
    for cat, idxs in selected_idx_dict.items():
        result = result + idxs
    return result

def split_by_labels(train_data, train_labels):
    '''
    将数据按照标签分类，以字典形式返回
    data_dict = {
        cat1: [trial1, trial2, trial3...],
        cat2: [trial4, trial7, trial8...],
    }
    '''
    unique_labels = sorted(set(train_labels)) # 全部标签列表
    data_dict = {}
    index_dict = {}
    for lab in unique_labels:
        idx = [i for i, l in enumerate(train_labels) if l == lab]
        data_dict[lab] = [train_data[i] for i in idx]
        index_dict[lab] = idx
    return data_dict, index_dict

def static_demo_predict(train_data, test_data, train_labels, test_labels, num_demos, max_predict_num, model_type, way_select_demo, is_model_online, test_num, combine_method, measurement, para, ablation, noise_target):
    data_dict, index_dict = split_by_labels(train_data, train_labels)
    num_cls = len(data_dict) # 类别数量

    # 下面的mean_dict和label_names均用于prompt中
    mean_dict = {}
    for cat, data in data_dict.items():
        mean_dict[cat] = np.mean(data, axis=0)

    label_names = sorted(set(train_labels))

    if way_select_demo == "BL":
        selected_idx_dict = {}
        for cat, indices in index_dict.items():
            selected_idx_dict[cat] = random.sample(indices, num_demos // num_cls) # 返回一个新的 list，包含 k 个不重复的元素。
        selected_indices = transformIndices(selected_idx_dict, combine_method)

    elif way_select_demo == "RD":
        selected_idx_dict = {}
        for cat, data in data_dict.items():
            local_selected_idx = rd_basic(data, n_initial=1, n_queries=num_demos // num_cls, measurement=measurement)
            global_selected_idx = [index_dict[cat][i] for i in local_selected_idx]
            selected_idx_dict[cat] = global_selected_idx
        selected_indices = transformIndices(selected_idx_dict, combine_method)

    elif way_select_demo == "CTD":
        selected_idx_dict = {}
        for cat, data in data_dict.items():
            local_selected_idx = find_centroids(data, num_demos // num_cls, measurement)
            global_selected_idx = [index_dict[cat][i] for i in local_selected_idx]
            selected_idx_dict[cat] = global_selected_idx
        selected_indices = transformIndices(selected_idx_dict, combine_method)

    elif way_select_demo == "SVM-far":
        selected_indices = balanced_svm(train_data, train_labels, num_demos) # 取离分界面远的点，只在计算相邻样本时用到了measurement，但影响不大就统一用ed

    elif way_select_demo == "SVM-near":
        selected_indices = svm_near(train_data, train_labels, num_demos) # 直接取距离SVM分界面最近的，不涉及measurement

    elif way_select_demo == "PURE":
        selected_indices = bal_simn_multi(train_data, train_labels, num_demos, num_cls, measurement=measurement)

    demo_data = [train_data[i] for i in selected_indices]
    demo_labels = [train_labels[i] for i in selected_indices]
    print(way_select_demo, selected_indices, demo_labels)
    # =============================================================================== #
    # Input perturbation
    # =============================================================================== #
    print('!!!', type(train_data), type(train_data[0]), train_data[0].shape)
    train_arr = np.stack(train_data, axis=0)     # (N, 4, 2)
    std_arr = train_arr.reshape(len(train_data), -1).std(axis=0)  # (8,)
    print('Perturbation Dim: ', noise_target)

    def add_noise_on_targets(data_list, std_arr, noise_target, alpha=1.0, seed=None):
        """
        data_list: List[np.ndarray], each (4,2)
        noise_target: list/np.ndarray of ints in [0,7], e.g., [0,1] for first freq band
        alpha: sigma = alpha * std_arr[idx]
        #   band 0 -> idx [0,1]
        #   band 1 -> idx [2,3]
        #   band 2 -> idx [4,5]
        #   band 3 -> idx [6,7]
        """
        rng = np.random.default_rng(seed)
        targets = np.array(noise_target, dtype=int).ravel()
        out = []
        for x in data_list:
            x2 = np.array(x, copy=True).reshape(-1)  # (8,)
            noise = rng.normal(0.0, alpha * std_arr[targets], size=targets.shape)
            x2[targets] += noise
            out.append(x2.reshape(4, 2))
        return out

    def mask_on_targets(data_list, noise_target, mask_token="[MASK]"):
        """
        data_list: List[np.ndarray], each (4,2)
        noise_target: list/np.ndarray of ints in [0,7]
        mask_token: str, e.g. "[MASK]" or "NA"
        """
        targets = np.array(noise_target, dtype=int).ravel()
        out = []

        for x in data_list:
            # 转成 object，允许数值 + 字符串共存
            x2 = np.array(x, copy=True, dtype=object).reshape(-1)  # (8,)
            for idx in targets:
                x2[idx] = mask_token
            out.append(x2.reshape(4, 2))

        return out
    

    def permute_band_features(data_list, noise_target, seed=None):
        """
        data_list: List[np.ndarray], each (4,2)
        noise_target: list/np.ndarray of ints in [0,7]
                    e.g., [4,5] means band=2 two CSP dims
        """
        rng = np.random.default_rng(seed)
        arr = np.stack(data_list, axis=0)        # (N, 4, 2)

        targets = np.array(noise_target, dtype=int).ravel()
        assert np.all((0 <= targets) & (targets <= 7)), f"noise_target out of range: {targets}"

        perm = rng.permutation(len(arr))         # (N,)
        flat = arr.reshape(len(arr), -1)         # (N, 8)
        flat[:, targets] = flat[perm][:, targets]
        arr2 = flat.reshape(len(arr), 4, 2)
        return [arr2[i] for i in range(len(arr2))]
    
    if ablation == 'feat_mask':
        print('feat_mask!!!')
        demo_data = mask_on_targets(demo_data, noise_target, mask_token="[MASK]")
    elif ablation == 'feat_noise':
        print('feat_noise!!!')
        demo_data = add_noise_on_targets(demo_data, std_arr, noise_target)
    elif ablation == 'feat_perm':
        print('feat_perm!!!')
        demo_data = permute_band_features(demo_data, noise_target)

    # =============================================================================== #
    # =============================================================================== #

    # =============================================================================== #
    # Example Ablation
    # =============================================================================== #
    def flip_half_labels(demo_labels, seed=None):
        """
        demo_labels: list，前半部分全为 A，后半部分全为 B
        返回：在前半(A)随机翻转一半为B；在后半(B)随机翻转一半为A
        """
        n = len(demo_labels)
        assert n % 2 == 0, "Expected even number of exemplars (e.g., 4-shot)."
        mid = n // 2

        A = demo_labels[0]
        B = demo_labels[-1]
        # 可选：强校验结构是否满足假设
        assert all(y == A for y in demo_labels[:mid]) and all(y == B for y in demo_labels[mid:]), \
            "demo_labels must be ordered: first half all A, second half all B."

        rng = np.random.default_rng(seed)
        out = list(demo_labels)

        # 前半(A)翻转一半
        k = mid // 2
        idx_left = rng.choice(np.arange(0, mid), size=k, replace=False)
        for i in idx_left:
            out[i] = B

        # 后半(B)翻转一半
        idx_right = rng.choice(np.arange(mid, n), size=k, replace=False)
        for i in idx_right:
            out[i] = A
        return out
    
    def flip_all_labels(demo_labels):
        """
        demo_labels: list，前半A后半B
        返回：所有 A<->B 全部互换
        """
        n = len(demo_labels)
        assert n % 2 == 0, "Expected even number of exemplars."
        mid = n // 2

        A = demo_labels[0]
        B = demo_labels[-1]
        assert all(y == A for y in demo_labels[:mid]) and all(y == B for y in demo_labels[mid:]), \
            "demo_labels must be ordered: first half all A, second half all B."
        return [B if y == A else A for y in demo_labels]
    
    if ablation == 'flip_half_labels':
        print('flip half labels!!!')
        demo_labels  = flip_half_labels(demo_labels)
    elif ablation == 'flip_all_labels':
        print('flip all labels!!!')
        demo_labels = flip_all_labels(demo_labels)
    # =============================================================================== #
    # =============================================================================== #

    if test_num:
        y_true = test_labels[:test_num]
        y_pred = collect_y_pred(demo_data, demo_labels, test_data[:test_num], max_predict_num, model_type, is_model_online, label_names, mean_dict, para)
    else:
        y_true = test_labels
        y_pred = collect_y_pred(demo_data, demo_labels, test_data, max_predict_num, model_type, is_model_online, label_names, mean_dict, para)

    accuracy, precision, recall, f1 = get_accuracy_and_log(y_true, y_pred)
    return accuracy, precision, recall, f1

def run_exp(exp_setting, dataset_name, sub_index, methods, test_id, repeat_times, num_demos, max_predict_num, model_type, is_model_online, test_num, combine_method, measurement, ablation, noise_target):
    if sub_index:
        sub_list = [sub_index]
    else:
        sub_list = list(range(1, num_subs[dataset_name] + 1))

    # 创建日志文件路径
    date_str = datetime.today().strftime("%Y-%m-%d")
    log_path = f'log/experiments/{exp_setting}_{dataset_name}_{date_str}.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding="utf-8") as f:
        # ===== 1. 记录参数 =====
        f.write('Experiment Parameters:\n')
        f.write(f'exp setting = {exp_setting}\n')
        f.write(f'dataset_name = {dataset_name}\n')
        f.write(f'sub_index = {sub_index}\n')
        f.write(f'methods = {methods}\n')
        f.write(f'test_id = {test_id}\n')
        f.write(f'repeat_times = {repeat_times}\n')
        f.write(f'num_demos = {num_demos}\n')
        f.write(f'max_predict_num = {max_predict_num}\n')
        f.write(f'model_type = {model_type}\n')
        f.write(f'is_model_online = {is_model_online}\n')
        f.write(f'test_num = {test_num}\n')
        f.write(f'combine_method = {combine_method}\n')
        f.write(f'measurement = {measurement}\n\n')

        for sub in sub_list:
            f.write("========================================\n")
            f.write(f"New Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"sub{sub}\n\n")
            f.flush()

            start_time = time.time()

            if exp_setting == 'within_sub':
                if dataset_name == 'seed':
                    train_data, test_data, train_labels, test_labels = get_seed_data(sub)
                    folds = [(train_data, test_data, train_labels, test_labels)]  # 统一封装成folds列表
                elif dataset_name == 'sleep-edfx':
                    X, y, _ = get_sleep_data(sub_indices=[sub]) # 改方法可返回单一被试也可返回多个被试
                    folds = []  # 五折
                    skf = StratifiedKFold(n_splits=5, shuffle=True)
                    for train_idx, test_idx in skf.split(X, y):
                        folds.append((X[train_idx], X[test_idx], y[train_idx], y[test_idx]))
                elif dataset_name == 'chb-mit':
                    data = np.load(f'data/chb-mit/chb_data_balanced_{sub}.npz')
                    X = data['X']
                    y = data['y']
                    y = y.astype(str)
                    folds = []  # 五折
                    skf = StratifiedKFold(n_splits=5, shuffle=True)
                    for train_idx, test_idx in skf.split(X, y):
                        folds.append((X[train_idx], X[test_idx], y[train_idx], y[test_idx]))
                else:
                    train_data, test_data, train_labels, test_labels = get_moabb_data(
                        dataset_name, sub, test_id, para_dict[dataset_name]
                    )
                    folds = [(train_data, test_data, train_labels, test_labels)]  # 统一封装成folds列表
            elif exp_setting == 'cross_sub':
                # 需根据范式判断获取数据的方法。跨被试暂时只支持mi和sleep两种范式
                dataset_para = para_dict[dataset_name]
                get_data_dict = {
                    'mi': get_moabb_data_crs_sub,
                    'sleep': get_sleep_data_crs_sub,
                    'epilepsy': get_ep_data,
                }
                get_data = get_data_dict[dataset_para]

                train_data, test_data, train_labels, test_labels = get_data(
                    dataset_name, sub, para_dict[dataset_name]
                ) # 跨被试
                folds = [(train_data, test_data, train_labels, test_labels)]  # 统一封装成folds列表

            result_dict = {method: [] for method in methods}
            for fold_idx, (train_data, test_data, train_labels, test_labels) in enumerate(folds):
                train_data = ndToList(train_data)
                test_data = ndToList(test_data)
                train_labels = train_labels.tolist()
                test_labels = test_labels.tolist()

                for n in tqdm(range(repeat_times)):
                    for method in methods:
                        acc, _, _, _ = static_demo_predict(
                            train_data, test_data, train_labels, test_labels,
                            num_demos, max_predict_num, model_type,
                            method, is_model_online, test_num,
                            combine_method, measurement, para_dict[dataset_name], ablation, noise_target
                        )
                        print(method, acc)
                        result_dict[method].append(acc)

                        # 立刻写入一组日志
                        with open(log_path, 'a') as logf:
                            logf.write(f"{method} - fold {fold_idx+1} - repeat {n+1}/{repeat_times}: {acc:.4f}\n")
                            logf.flush()

            # 写每个方法的准确率列表（完整收集）
            for method, acc_list in result_dict.items():
                acc_str = ', '.join([f"{x:.4f}" for x in acc_list])
                f.write(f"{method}: [{acc_str}]\n")
            f.write('\n')

            # 写平均值和标准差
            f.write("平均值\n")
            for method, acc_list in result_dict.items():
                acc_array = np.array(acc_list)
                mean_val = acc_array.mean()
                std_val = acc_array.std()
                f.write(f"{method}: {mean_val:.4f}, {std_val:.4f}\n")
            f.write('\n')

            end_time = time.time()
            elapsed = end_time - start_time
            f.write(f"单个被试耗时：{elapsed:.1f}s\n\n")
            f.flush()

if __name__ == '__main__':
    para_dict = {
        '2a': 'mi',
        '2b': 'mi',
        'BNCI2015_001': 'mi',
        'BNCI2014_002': 'mi',
        'Weibo2014': 'mi',
        'sleep-edfx': 'sleep',
        'TUEP': 'epilepsy',
        'chb-mit': 'epilepsy',
        'iris': 'uci',
        'wine': 'uci',
        'wdbc': 'uci',
        'glass': 'uci',
        'seed': 'emo',
    }
    num_subs = {
        '2a': 9,
        '2b': 9,
        'BNCI2015_001': 12,
        'BNCI2014_002': 14,
        'Weibo2014': 10,
        'sleep-edfx': 7,
        'TUEP': 10,
        'chb-mit': 1,
        'iris': 1,
        'wine': 1,
        'wdbc': 1,
        'glass': 1,
        'seed': 15,
    }

    args = get_args()
    print('入参:', args)

    # run_exp(exp_setting, dataset_name, sub_index, methods, test_id, repeat_times, num_demos, max_predict_num, model_type, is_model_online, test_num, combine_method, measurement)
    run_exp(**vars(args))


    # num_demos = 4 # 演示示例的数量
    # sub_index = 6 # 被试编号(1-9)
    # max_predict_num = 10 # 单次最多预测样本的个数，演示示例+单次预测样本个数，加起来的本文长度不能超过LLM的max_token
    # model_type = "qwen2.5-14b-instruct-1m" # "qwen2.5-14b-instruct-1m", "deepseek-chat", "deepseek-reasoner", "qwen-long" "qwen2.5-3b-instruct" "Qwen/Qwen2.5-Coder-32B-Instruct" # 'Qwen/Qwen2.5-7B-Instruct'# "qwen2.5-7b-instruct" Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-1.5B-Instruct
    # dataset_name = "BNCI2014_002" # 2a, 2b, BNCI2015_001, BNCI2014_002, Nakanishi2015, BNCI2014008, iris, wine, TUEP
    # test_id = 1 # 2a中只有0-1两个session，2b中有0-4五个session, BNCI2015_001中两个session
    # is_model_online = True # 谨慎开启，设置为online时要提前计算费用
    # measurement = 'ed' # 衡量向量距离的方式，ed, cs
    # combine_method = None
    # repeat_times = 30
    # test_num = None # 10, None

    # today = datetime.today()
    # date_str = today.strftime("%Y-%m-%d")

    # # 5-folds
    # log_filename = f'log/log_acc_5_folds_{dataset_name}_{date_str}.txt'
    # log_lists = {1: [], 2: [], 3: []}  # 存储所有被试的精度，用于最终汇总
    # dataset_para = para_dict[dataset_name]
    # get_data_cv_dict = {
    #     'uci': get_uci_data_cv,
    #     'mi': get_moabb_data_cv,
    #     'sleep': get_sleep_data_cv,
    #     'epilepsy': get_ep_data_cv,
    # }
    # get_data_cv = get_data_cv_dict[dataset_para]

    # with open(log_filename, "a") as log_file:
    #     log_file.write("\n" + "=" * 50 + "\n")
    #     log_file.write(f"5-Fold Cross Validation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # for sub_index in range(1, num_subs[dataset_name] + 1):  # 被试编号范围
    #     list1, list2, list3 = [], [], []

    #     with open(log_filename, "a") as log_file:
    #         log_file.write("\n" + "-" * 30 + f"\nSubject {sub_index}\n")
    #         log_file.write("Fold\tAcc1\tAcc2\tAcc3\tMean1\tStd1\tMean2\tStd2\tMean3\tStd3\n")

    #     for fold_idx, (train_X, test_X, train_y, test_y) in enumerate(
    #         tqdm(get_data_cv(dataset_name, sub_index, para_dict[dataset_name]), total=5, desc=f"CV for {dataset_name}-sub{sub_index}")
    #     ):
    #         train_data = ndToList(train_X)
    #         test_data = ndToList(test_X)
    #         train_labels = train_y.tolist()
    #         test_labels = test_y.tolist()
    #         print('测试集标签:', test_labels)

    #         accuracy1, _, _, _ = static_demo_predict(train_data, test_data, train_labels, test_labels, num_demos, max_predict_num, model_type, 'random', is_model_online, test_num, combine_method)
    #         accuracy2, _, _, _ = static_demo_predict(train_data, test_data, train_labels, test_labels, num_demos, max_predict_num, model_type, 'rd_basic', is_model_online, test_num, combine_method)
    #         accuracy3, _, _, _ = static_demo_predict(train_data, test_data, train_labels, test_labels, num_demos, max_predict_num, model_type, 'bal_simn_multi', is_model_online, test_num, combine_method)

    #         list1.append(accuracy1)
    #         list2.append(accuracy2)
    #         list3.append(accuracy3)

    #         mean1, std1 = np.mean(list1), np.std(list1)
    #         mean2, std2 = np.mean(list2), np.std(list2)
    #         mean3, std3 = np.mean(list3), np.std(list3)

    #         with open(log_filename, "a") as log_file:
    #             log_file.write(f"{fold_idx+1}\t{accuracy1:.4f}\t{accuracy2:.4f}\t{accuracy3:.4f}\t{mean1:.4f}\t{std1:.4f}\t{mean2:.4f}\t{std2:.4f}\t{mean3:.4f}\t{std3:.4f}\n")
    #             # log_file.write(f"{fold_idx+1}\t{accuracy1:.4f}\t{mean1:.4f}\t{std1:.4f}\n")
    #             log_file.flush()

    #     # 记录每个被试最终的平均结果
    #     log_lists[1].extend(list1)
    #     log_lists[2].extend(list2)
    #     log_lists[3].extend(list3)

    # # ============ 最终结果汇总 ============

    # with open(log_filename, "a") as log_file:
    #     log_file.write("\n" + "=" * 30 + "\n最终平均结果：\n")
    #     for i in range(1, 4):
    #         all_vals = log_lists[i]
    #         if all_vals:
    #             mean_val = np.mean(all_vals)
    #             std_val = np.std(all_vals)
    #             log_file.write(f"List {i} (method {i}): 平均 = {mean_val:.4f}，标准差 = {std_val:.4f}\n")
    #         else:
    #             log_file.write(f"List {i} 是空的，无法计算\n")
    #     log_file.write("=" * 50 + "\n")





