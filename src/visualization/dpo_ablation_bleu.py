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

Qwen_no_SV = [
    [4.30, 9.39, 4.78, 9.81, 4.48, 10.11, 3.16, 11.96],
    [10.38, 5.99, 3.06, 11.04, 6.25, 8.08, 1.91, 12.67],
    [6.68, 8.05, 3.92, 10.64, 4.12, 9.58, 1.86, 12.51]
]
Qwen_no_cluster = [
    [18.17, 4.07, 20.80, 4.09, 12.52, 4.39, 24.86, 3.97],
    [18.40, 4.01, 21.20, 4.03, 12.57, 4.39, 24.53, 3.98],
    [18.24, 3.97, 20.45, 4.15, 12.49, 4.41, 24.99, 3.95]
]
Qwen_ours = [
    [18.04, 3.87, 20.34, 4.01, 11.93, 4.41, 23.46, 4.03],
    [17.31, 4.09, 19.97, 4.13, 11.62, 4.41, 22.49, 4.26],
    [17.42, 3.99, 20.32, 4.07, 11.68, 4.45, 22.86, 4.21]
]
Llama_no_SV = [
    [1.04, 10.40, 0.61, 11.79, 0.90, 11.51, 4.40, 11.04],
    [1.70, 9.92, 1.22, 11.15, 0.73, 11.91, 2.91, 11.40],
    [1.09, 10.77, 1.04, 11.42, 0.61, 12.30, 2.34, 11.95]
]
Llama_no_cluster = [
    [2.43, 10.21, 5.30, 9.47, 2.06, 10.98, 6.91, 9.92],
    [3.12, 9.71, 5.05, 9.52, 2.31, 10.57, 5.87, 10.64],
    [2.45, 10.01, 3.72, 10.49, 2.45, 10.56, 5.49, 10.68]
]
Llama_ours = [
    [5.28, 8.43, 5.59, 9.27, 2.75, 9.99, 6.89, 9.70],
    [4.73, 8.85, 5.33, 9.41, 2.96, 9.82, 7.27, 9.64],
    [4.74, 8.98, 5.07, 9.58, 3.16, 9.71, 7.00, 9.65]
]
total_data = [
    ("Qwen-1.5B", "SVSI", cal_relative(Qwen_ours, Qwen_base)),
    ("Qwen-1.5B", "w/o. clustering", cal_relative(Qwen_no_cluster, Qwen_base)),
    ("Qwen-1.5B", "w/o. SV", cal_relative(Qwen_no_SV, Qwen_base)),
    ("Llama-1B", "SVSI", cal_relative(Llama_ours, Llama_base)),
    ("Llama-1B", "w/o. clustering", cal_relative(Llama_no_cluster, Llama_base)),
    ("Llama-1B", "w/o. SV", cal_relative(Llama_no_SV, Llama_base)),
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
legend = axes[0].legend(
    ncol=3,                           # 横向3列
    loc='upper center',               # 位置
    bbox_to_anchor=(0.5, 1.25),       # 微调位置避免遮挡
    fontsize=22,                      # 字体大小
    title_fontsize=23,                # 标题字体大小
    frameon=False                     # 可选：去掉边框更清爽
)

    
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
plt.savefig("dpo_ablation", dpi=144, bbox_inches='tight')
