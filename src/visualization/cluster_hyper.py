import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogFormatter
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')
sns.set_theme(style="whitegrid", font="Times New Roman")
plt.rcParams.update({
    'font.size': 14,           # 默认文本字体大小
    'axes.labelsize': 24,      # 坐标轴标签字体大小
    'axes.titlesize': 18,      # 子图标题字体大小
    'xtick.labelsize': 18,     # x轴刻度字体大小
    'ytick.labelsize': 18,     # y轴刻度字体大小
    'legend.fontsize': 24,     # 图例字体大小
    'figure.titlesize': 24     # 整个图像的标题大小（如果有）
})

map_mqm = lambda x: (15 - x) / 15

def cal_relative(arr, base_num, mapping = False):
    if mapping:
        return [(map_mqm(num) - map_mqm(base_num))/map_mqm(base_num) * 100 for num in arr]
    else:
        return [(num - base_num)/base_num * 100 for num in arr]

Q1B_bleu = [14.20, 13.67, 14.23, 15.78, 16.00, 15.84, 14.33, 18.34, 18.04, 17.36, 16.50, 17.04, 17.70, 18.11, 17.97, 17.66, 18.35, 18.23]
Q1B_MQM = [4.79, 5.06, 4.86, 4.56, 4.25, 4.25, 4.78, 3.84, 3.87, 3.90, 4.04, 3.93, 3.87, 3.88, 3.82, 3.87, 3.82, 3.84]
Q3B_bleu = [18.83, 19.86, 19.43, 19.20, 19.39, 19.14, 19.92, 20.02, 20.44, 19.99, 19.94, 20.40, 20.41, 20.43, 20.21, 20.20, 20.34, 19.93]
Q3B_MQM = [4.01, 3.76, 3.78, 3.87, 3.90, 3.88, 3.69, 3.73, 3.66, 3.71, 3.78, 3.70, 3.70, 3.65, 3.71, 3.75, 3.74, 3.77]
Q7B_bleu = [19.75, 20.09, 19.87, 20.01, 20.07, 19.77, 19.36, 20.48, 20.19, 20.40, 19.87, 19.22, 20.11, 20.81, 20.46, 20.03, 19.80, 19.55]
Q7B_MQM = [4.27, 4.08, 4.18, 4.15, 4.09, 4.15, 4.29, 4.14, 4.23, 4.11, 4.20, 4.34, 4.21, 3.80, 4.06, 3.99, 4.04, 4.36]

total_data = [
    cal_relative(Q1B_bleu, 16.12),
    cal_relative(Q1B_MQM, 4.23, mapping=True),
    cal_relative(Q3B_bleu, 17.64),
    cal_relative(Q3B_MQM, 3.97, mapping=True),
    cal_relative(Q7B_bleu, 16.43),
    cal_relative(Q7B_MQM, 5.10, mapping=True),
]
models = ["Qwen-1.5B", "Qwen-3B", "Qwen-7B"]
pd_data = {
    "min_cluster_size": [],
    "min_samples": [],
    "improvements (%)":[],
    "metric": [],
    "model": []
}
for i in range(len(total_data)):
    num_arr = total_data[i]
    if i % 2 == 0:
        metric = "BLEU"
    else:
        metric = "MQM"
    model = models[i // 2]
    cnt = 0
    for min_cluster_size in [3,4,5,6]:
        for min_samples in range(1, min_cluster_size+1):
            value = num_arr[cnt]
            pd_data["min_cluster_size"].append(min_cluster_size)
            pd_data["min_samples"].append(min_samples)
            pd_data["improvements (%)"].append(value)
            pd_data["metric"].append(metric)
            pd_data["model"].append(model)
            cnt += 1
custom_palette = ["#B85450", "#6C8EBF", "#82B366", "#D6B656"]
g = sns.relplot(
    data=pd_data,
    x="min_samples", y="improvements (%)", hue="min_cluster_size",
    col="model", row="metric",
    kind="line", markers=True,
    dashes=False,
    linewidth = 4, markersize=15, style="min_cluster_size",
    facet_kws={'sharey': False, 'sharex': True},
    height=4, aspect=1.8,
    palette=custom_palette
)
if g._legend:
    # 设置图例标题字体大小
    g._legend.get_title().set_fontsize(22) 

plt.savefig("cluster_hyper.png")
