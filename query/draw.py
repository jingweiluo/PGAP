import sys
import os
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import itertools
from active_prompt.load.load_moabb import get_moabb_data

def plot_tsne_with_all_indices(train_data, train_labels, sample_map, sub_index):
    """
    对 144x8 的数据进行 t-SNE 降维，并在图中标注所有点的索引。

    参数：
    - train_data: ndarray, 形状 (144, 8)，输入数据
    - train_labels: ndarray, 形状 (144,)，标签，值为 'left_hand' 或 'right_hand'
    - sample_map: dict, 包含多个索引类别的字典
    """
    sns.set(style="whitegrid")  # 使用 Seaborn 美化绘图风格

    train_data = np.array(train_data)
    data = train_data.reshape(train_data.shape[0], -1)
    labels = np.array(train_labels)

    # 进行 t-SNE 降维到 2 维
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    # 生成高对比度的颜色
    num_classes = len(set(labels))  # 获取类别数量
    colors = sns.color_palette("husl", num_classes)  # 使用 Seaborn 的 HUSL 颜色空间生成颜色
    color_map = {label: colors[i] for i, label in enumerate(set(labels))}
    colors = [(color[0], color[1], color[2], 0.2) for label, color in zip(labels, [color_map[label] for label in labels])]

    # 生成不同类别的高亮颜色和形状
    palette = sns.color_palette("husl", len(sample_map))
    highlight_colors = {category: palette[i] for i, category in enumerate(sample_map.keys())}
    markers = ['s', 'D', '^', 'v', '<', '>', 'p', '*', 'X']
    highlight_markers = {category: markers[i % len(markers)] for i, category in enumerate(sample_map.keys())}

    # 绘制降维后的数据点（类别颜色，透明度低）
    plt.figure(figsize=(12, 9))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, alpha=0.2, s=150, edgecolors='black', label="Samples")

    # # 在所有点上标注索引
    # for idx in range(len(data_2d)):
    #     plt.text(data_2d[idx, 0] + 0.2, data_2d[idx, 1] + 0.2, str(idx), fontsize=8, color='black', weight='bold')

    # 处理高亮点，支持多个标签
    legend_handles = []
    plotted_points = {}
    for category, indices in sample_map.items():
        for idx in indices:
            if idx not in plotted_points:
                plotted_points[idx] = []
            plotted_points[idx].append((highlight_colors[category], highlight_markers[category]))

    for idx, highlight_list in plotted_points.items():
        for i, (color, marker) in enumerate(highlight_list):
            offset_x = (i % 3 - 1) * 0.05
            offset_y = (i // 3 - 1) * 0.05
            plt.scatter(data_2d[idx, 0] + offset_x, data_2d[idx, 1] + offset_y,
                        c=[color], marker=marker, edgecolors='black', s=80, linewidth=1)

    # 添加类别对应颜色的注释
    for category, color in color_map.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=(color[0], color[1], color[2], 0.6), markersize=10, 
                                         label=f"Class: {category}"))

    # 添加高亮类别的图例
    for category, color in highlight_colors.items():
        legend_handles.append(plt.Line2D([0], [0], marker=highlight_markers[category], color='w', 
                                         markerfacecolor=color, markersize=10, label=f"Highlight: {category}"))

    # 图例
    plt.legend(handles=legend_handles, title="Categories and Highlights", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Visualization of subject {sub_index}", fontsize=14)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)

    # 保存并显示图像
    plt.tight_layout()
    plt.savefig("tsne_visualization.png", dpi=300)
    plt.savefig("tsne_visualization.eps", dpi=300, format='eps')
    plt.show()

# dataset_name = "2a"
# sub_index = 6
# test_id = 1
# k = 4

# train_data, test_data, train_labels, test_labels = get_moabb_data(dataset_name, sub_index, test_id)
# plot_tsne_with_all_indices(train_data, train_labels, [96,133,141])
