import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')
sns.set_theme(style="whitegrid", font="Times New Roman")
plt.rcParams.update({
    'font.size': 14,           # 默认文本字体大小
    'axes.labelsize': 30,      # 坐标轴标签字体大小
    'axes.titlesize': 25,      # 子图标题字体大小
    'xtick.labelsize': 24,     # x轴刻度字体大小
    'ytick.labelsize': 24,     # y轴刻度字体大小
    'legend.fontsize': 30,     # 图例字体大小
    'figure.titlesize': 30     # 整个图像的标题大小（如果有）
})

map_mqm = lambda x: 1/(1+x)

def cal_relative(arr, base):
    # for i in range(len(arr)):
    #     for j in range(len(arr[i])):
    #         if j % 2 == 1:
    #             arr[i][j] = - arr[i][j]
    # return arr
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if j % 2 == 0:
                # arr[i][j] = (arr[i][j] - base[j]) / base[j] * 100
                arr[i][j] = arr[i][j]
            else:
                # arr[i][j] = (map_mqm(arr[i][j]) - map_mqm(base[j])) / map_mqm(base[j]) * 100
                arr[i][j] = map_mqm(arr[i][j]) * 100
    return arr
    
def subset(pd_data, metric, model):
    subset_data = {
        "Dataset": [],
        "Improvement (%)": [],
        "Method": []
    }
    for i in range(len(pd_data["Metric"])):
        if pd_data["Metric"][i] == metric and pd_data["Model"][i] == model:
            subset_data["Dataset"].append(pd_data["Dataset"][i])
            subset_data["Improvement (%)"].append(pd_data["Improvement (%)"][i])
            subset_data["Method"].append(pd_data["Method"][i])
    return subset_data

Qwen_base = [16.12, 4.23, 13.63, 6.29, 10.63, 4.91, 20.11, 4.96]
Llama_base = [2.91, 9.65, 4.29, 9.99, 1.71, 11.09, 5.47, 10.30]

Qwen_empo = [
    [15.95, 4.19, 13.77, 6.13, 11.06, 4.73, 20.16, 4.90],
    [15.96, 4.22, 13.77, 6.18, 10.79, 4.72, 19.92, 5.08],
    [16.15, 4.19, 14.06, 5.99, 11.04, 4.68, 20.68, 4.61]
]
Qwen_entropy = [
    [8.12, 7.47, 6.97, 8.76, 5.72, 7.27, 8.81, 9.30],
    [10.35, 6.23, 13.42, 6.33, 11.10, 5.31, 7.69, 9.50],
    [13.91, 4.08, 4.67, 9.83, 6.47, 6.88, 7.03, 9.52]
]
Qwen_ours = [
    [17.96, 3.89, 14.59, 6.19, 11.06, 4.72, 22.36, 4.41],
    [17.95, 3.88, 15.76, 5.84, 11.13, 4.96, 20.44, 4.87],
    [16.39, 4.13, 15.34, 5.99, 11.80, 4.54, 22.10, 4.45]
]
Llama_empo = [
    [2.97, 9.81, 3.72, 10.17, 1.40, 11.43, 5.03, 10.81],
    [2.57, 9.92, 3.86, 10.14, 1.41, 11.47, 4.88, 10.84],
    [2.87, 9.79, 3.91, 10.26, 1.48, 11.53, 4.95, 10.81]
]
Llama_entropy = [
    [1.04, 10.72, 1.69, 11.52, 0.48, 12.71, 1.01, 12.96],
    [0.96, 10.89, 1.53, 11.65, 0.39, 12.69, 1.59, 12.60],
    [0.87, 10.92, 1.34, 11.53, 0.34, 12.65, 1.23, 12.73]
]
Llama_ours = [
    [3.32, 9.55, 5.51, 9.57, 1.72, 11.25, 5.02, 10.86],
    [3.38, 9.55, 4.65, 9.71, 1.54, 11.27, 5.83, 10.52],
    [3.42, 9.55, 4.89, 9.68, 1.60, 11.27, 5.31, 10.51]
]
total_data = [
    ("Qwen-1.5B", "Ours", cal_relative(Qwen_ours, Qwen_base)),
    ("Qwen-1.5B", "EMPO", cal_relative(Qwen_empo, Qwen_base)),
    ("Qwen-1.5B", "Entropy", cal_relative(Qwen_entropy, Qwen_base)),
    ("Llama-1B", "Ours", cal_relative(Llama_ours, Llama_base)),
    ("Llama-1B", "EMPO", cal_relative(Llama_empo, Llama_base)),
    ("Llama-1B", "Entropy", cal_relative(Llama_entropy, Llama_base)),
]
pd_data = {
    "Model": [],
    "Method": [],
    "Dataset": [],
    "Improvement (%)": [],
    "Metric": []
}
dataset_list = ["wmt24pp_de", "wmt24pp_fr", "wmt24pp_ru", "wmt24pp_es"]
for model_name, method_name, data in total_data:
    for i in range(len(data)):
        for j in range(len(data[i])):
            pd_data["Model"].append(model_name)
            pd_data["Method"].append(method_name)
            pd_data["Dataset"].append(dataset_list[j//2])
            pd_data["Improvement (%)"].append(data[i][j])
            if j % 2 == 0:
                pd_data["Metric"].append("BLEU")
            else:
                pd_data["Metric"].append("n-MQM")
inner_palette = ["#FAD9D5", "#B0E3E6", "#D0CEE2"]
edge_palette = ["#AE4132", "#0E8088", "#56517E"]

# fig, axes = plt.subplots(2, 2, figsize=(24, 8))
# columns = ["BLEU", "MQM"]
# rows = ["Qwen-1.5B", "Llama-1B"]
# for i in range(2):
#     for j in range(2):
#         ax = axes[i][j]
#         ax_metric = columns[j]
#         ax_model = rows[i]
#         pd_data_subset = subset(pd_data, ax_metric, ax_model)
#         sns.barplot(
#             data=pd_data_subset,
#             x="Dataset",
#             y="Improvement (%)",
#             hue="Method",
#             ax=ax,
#             errorbar="se",
#             palette=inner_palette,
#             edgecolor=".2",
#             errcolor=".2",
#             errwidth=1.5,
#             capsize=0.1,
#         )
#         ax.set_title(f"{rows[i]} - {columns[j]}")
#         if ax.get_legend():
#             ax.get_legend().remove()
#         ax.axhline(0, ls="--", color="gray", lw=1)
#         ax.set_ylim(-60, 160)
#         if j == 0:
#             ax.set_ylabel("Improvement (%)")
#         else:
#             ax.set_ylabel("")
#         if i == 1:
#             ax.set_xlabel("Dataset")
#         else:
#             ax.set_xlabel("")
#         for patch in ax.patches:
#             patch.set_edgecolor(edge_palette[(list(pd_data["Method"]).index(patch.get_facecolor()) % 3)])


g = sns.catplot(
    data=pd_data,
    x="Dataset",
    y="Improvement (%)",
    hue="Method",
    col="Model",
    row="Metric",
    kind="bar",
    errorbar="sd",
    height=4,
    aspect=3.5,
    palette=inner_palette,
    legend_out=True,
    sharey=False,
    sharex=True,
    gap=0.2,
)
# g.set(yscale="symlog")
# if g._legend:
#     # 设置图例标题字体大小
#     g._legend.get_title().set_fontsize(28) 
if g._legend:
    handles = g._legend.legend_handles
    labels = [text.get_text() for text in g._legend.get_texts()]
    g._legend.remove()
# 获取第一个轴的句柄和标签（所有子图共享相同 hue，所以取第一个即可）

# 在整个图的顶部添加横向图例
g.figure.legend(
    handles, 
    labels, 
    loc='upper center',        # 图例位置：顶部居中
    bbox_to_anchor=(0.45, 1.12), # 相对于整个图的位置，1.05 在图上方一点
    ncol=len(labels),          # 横向排列，每个标签一列
    fontsize=28,               # 字体大小
    title_fontsize=28          # 标题字体大小
)
cnt = 0
for ax in g.axes.flat:
    for patch in ax.patches:
        if cnt >= 48:
            continue
        patch.set_edgecolor(edge_palette[(cnt % 12) // 4])
        cnt += 1
g.tight_layout()
plt.savefig("grpo_ablation.png", dpi=144, bbox_inches='tight')