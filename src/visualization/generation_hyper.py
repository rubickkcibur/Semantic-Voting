import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogFormatter
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

L1 = [
    [5.08, 8.86, 5.05, 9.07, 2.32, 10.17, 2.85, 9.83, 0, 25, 0, 25],
    [3.74, 9.39, 3.94, 9.35, 3.04, 9.69, 3.57, 9.64, 0, 25, 0, 25],
    [2.33, 10.21, 5.28, 8.43, 3.13, 9.72, 2.24, 9.90, 0, 25, 0, 25],
    [2.70, 9.74, 3.84, 9.47, 2.96, 9.71, 2.75, 9.88, 0, 25, 0, 25]
]
L3 = [
    [6.96, 8.21, 6.91, 8.38, 8.50, 7.74, 7.29, 8.23, 0, 25, 0, 25],
    [10.44, 7.02, 7.83, 7.70, 9.41, 7.41, 7.33, 8.12, 7.81, 7.86, 0, 25],
    [9.37, 7.10, 8.71, 7.63, 10.05, 7.06, 7.72, 7.83, 7.77, 7.91, 0, 25],
    [9.84, 7.02, 9.29, 7.34, 6.72, 8.42, 6.55, 8.32, 7.04, 8.16, 0, 25],
]
Q1 = [
    [16.12, 4.14, 17.65, 3.88, 17.64, 3.94, 18.57, 3.88, 16.83, 4.08, 0, 25],
    [16.81, 4.06, 16.57, 4.04, 17.95, 3.85, 17.95, 3.93, 17.17, 4.05, 0, 25],
    [16.00, 4.29, 18.04, 3.87, 17.18, 3.84, 17.24, 3.97, 16.97, 4.09, 15.55, 4.20],
    [15.74, 4.29, 16.46, 4.27, 18.13, 3.97, 16.95, 4.05, 17.09, 4.07, 16.02, 4.25]
]
Q3 = [
    [18.84, 3.96, 20.45, 3.73, 20.55, 3.65, 20.35, 3.67, 19.71, 3.72, 19.44, 3.67],
    [19.74, 3.90, 20.01, 3.73, 20.24, 3.69, 19.28, 3.77, 19.47, 3.65, 19.61, 3.69],
    [20.00, 3.67, 20.44, 3.66, 20.07, 3.67, 18.37, 3.80, 17.46, 3.98, 19.56, 3.65],
    [19.79, 3.75, 20.43, 3.71, 19.14, 3.84, 18.68, 3.74, 18.27, 3.79, 19.46, 3.65]
]

total_data = {
    "Llama-1B": (L1, 2.91, 9.65),
    "Qwen-1.5B": (Q1, 16.12, 4.23),
    "Llama-3B": (L3, 6.35, 8.22),
    "Qwen-3B": (Q3, 17.64, 3.97),
}


models = total_data.keys()

pd_data = {
    "Temperature": [],
    "Sampling Size": [],
    "Improvements (%)":[],
    "Metric": [],
    "Model": []
}
temp_list = [0.3, 0.7, 1.0, 1.2, 1.5, 2.0]
size_list = [16, 32, 64, 128]
for model_name, (data, base_BLEU, base_MQM) in total_data.items():
    for i in range(len(data)):
        for j in range(len(data[i])):
            if j % 2 == 0:
                improve = (data[i][j] - base_BLEU) / base_BLEU * 100
                # improve = data[i][j]
            else:
                improve = (map_mqm(data[i][j]) - map_mqm(base_MQM)) / map_mqm(base_MQM) * 100
                # improve = map_mqm(data[i][j])
            pd_data["Temperature"].append(temp_list[j // 2])
            pd_data["Sampling Size"].append(size_list[i])
            pd_data["Improvements (%)"].append(improve)
            pd_data["Metric"].append("BLEU" if j % 2 == 0 else "n-MQM")
            pd_data["Model"].append(model_name)

custom_palette = ["#B85450", "#6C8EBF", "#82B366", "#D6B656"]
g = sns.relplot(
    data=pd_data,
    x="Temperature", y="Improvements (%)", hue="Sampling Size",
    col="Model", row="Metric",
    kind="line", markers=True,
    dashes=False,
    linewidth = 4, markersize=18, style="Sampling Size",
    facet_kws={'sharey': False, 'sharex': True},
    height=4, aspect=1.8,
    palette=custom_palette
)
g.set(yscale="symlog")
if g._legend:
    # 设置图例标题字体大小
    g._legend.get_title().set_fontsize(30) 
g.tight_layout()
plt.savefig("generation_hyper.png", dpi=144)
