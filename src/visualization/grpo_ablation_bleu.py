import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')
sns.set_theme(style="whitegrid", font="Times New Roman")
# plt.rcParams.update({
#     'font.size': 14,           # 默认文本字体大小
#     'axes.labelsize': 30,      # 坐标轴标签字体大小
#     'axes.titlesize': 25,      # 子图标题字体大小
#     'xtick.labelsize': 24,     # x轴刻度字体大小
#     'ytick.labelsize': 24,     # y轴刻度字体大小
#     'legend.fontsize': 30,     # 图例字体大小
#     'figure.titlesize': 30     # 整个图像的标题大小（如果有）
# })

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
    ("Qwen-1.5B", "SVSI-G", cal_relative(Qwen_ours, Qwen_base)),
    ("Qwen-1.5B", "EMPO", cal_relative(Qwen_empo, Qwen_base)),
    ("Qwen-1.5B", "EMRL-seq", cal_relative(Qwen_entropy, Qwen_base)),
    ("Llama-1B", "SVSI-G", cal_relative(Llama_ours, Llama_base)),
    ("Llama-1B", "EMPO", cal_relative(Llama_empo, Llama_base)),
    ("Llama-1B", "EMRL-seq", cal_relative(Llama_entropy, Llama_base)),
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

sns.set_theme(style="whitegrid", font="Times New Roman")
inner_palette = ["#FAD9D5", "#B0E3E6", "#D0CEE2"]
edge_palette = ["#AE4132", "#0E8088", "#56517E"]
hatches = ["x", "/", "."]
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

qwen_data = {
    "Dataset": [],
    "Method": [],
    "BLEU": []
}
for i in range(len(pd_data["Model"])):
    model = pd_data["Model"][i]
    method = pd_data["Method"][i]
    dataset = pd_data["Dataset"][i]
    metric = pd_data["Metric"][i]
    improvement = pd_data["Improvement (%)"][i]
    if model == "Qwen-1.5B" and metric == "BLEU":
        qwen_data["Dataset"].append(dataset)
        qwen_data["Method"].append(method)
        qwen_data["BLEU"].append(improvement)
        
g = sns.barplot(
    data=qwen_data,
    x="Dataset", y="BLEU", hue="Method", 
    palette=inner_palette, 
    # edgecolor=sns.color_palette(edge_palette),
    alpha=1.0, linewidth=2,
    width=0.5, errorbar="sd",
    legend="full",
    capsize=0.2,
    err_kws={'linewidth': 2},
    ax=axes[0]
)
# axes[0].set_title("Qwen-1.5B", size=30)
axes[0].set_xlabel("Qwen-1.5B", size=30)
axes[0].set_ylabel("BLEU", size=30)
axes[0].tick_params(axis='both', which='major', labelsize=24)
cnt = 0
for bar in g.patches:
    if cnt >= 12:
        continue
    bar.set_edgecolor(edge_palette[cnt//4])
    bar.set_hatch(hatches[cnt//4])
    cnt+=1
# legend = axes[0].legend(
#     ncol=3,                           # 横向3列
#     loc='upper center',               # 位置
#     bbox_to_anchor=(0.5, 1.25),       # 微调位置避免遮挡
#     fontsize=22,                      # 字体大小
#     title_fontsize=23,                # 标题字体大小
#     frameon=False                     # 可选：去掉边框更清爽
# )

llama_data = {
    "Dataset": [],
    "Method": [],
    "BLEU": []
}
for i in range(len(pd_data["Model"])):
    model = pd_data["Model"][i]
    method = pd_data["Method"][i]
    dataset = pd_data["Dataset"][i]
    metric = pd_data["Metric"][i]
    improvement = pd_data["Improvement (%)"][i]
    if model == "Llama-1B" and metric == "BLEU":
        llama_data["Dataset"].append(dataset)
        llama_data["Method"].append(method)
        llama_data["BLEU"].append(improvement)

g = sns.barplot(
    data=llama_data,
    x="Dataset", y="BLEU", hue="Method", 
    palette=inner_palette, 
    # edgecolor=sns.color_palette(edge_palette), 
    alpha=1.0, linewidth=2,
    width=0.5, errorbar="sd",
    legend=False,
    capsize=0.2,
    err_kws={'linewidth': 2},
    ax=axes[1]
)
# axes[1].set_title("Llama-1B", size=30)
axes[1].set_xlabel("Llama-1B", size=30)
axes[1].set_ylabel("BLEU", size=30)
axes[1].tick_params(axis='both', which='major', labelsize=24)
cnt = 0
for bar in g.patches:
    if cnt >= 12:
        continue
    bar.set_edgecolor(edge_palette[cnt//4])
    bar.set_hatch(hatches[cnt//4])
    cnt+=1

handles, labels = axes[0].get_legend_handles_labels()
new_handles = []
for i in range(3):
    handle = plt.Rectangle(
        (0, 0), 1, 1, 
        facecolor=inner_palette[i],  # 保持和柱子填充色一致
        edgecolor=edge_palette[i],
        hatch=hatches[i],
        linewidth=2
    )
    new_handles.append(handle)
axes[0].get_legend().remove()
fig.legend(
    new_handles, 
    labels, 
    frameon=False,
    markerscale=3,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    fontsize=20,
    title_fontsize=20
)

plt.tight_layout()
plt.savefig("grpo_ablation", dpi=144, bbox_inches='tight')
