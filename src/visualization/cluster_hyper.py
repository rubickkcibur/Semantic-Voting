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
L1B_bleu = [2.26, 3.17, 2.99, 3.33, 3.30, 4.56, 3.21, 4.92, 5.28, 4.95, 4.33, 4.28, 5.28, 5.23, 5.97, 4.73, 5.13, 4.06]
L1B_MQM = [10.22, 9.71, 9.82, 9.55, 9.54, 9.00, 9.73, 8.51, 8.43, 8.77, 9.08, 8.97, 8.78, 8.59, 8.53, 8.88, 8.87, 8.99]
L3B_bleu = [7.33, 6.42, 7.18, 7.48, 6.04, 8.44, 9.15, 7.96, 8.71, 8.50, 8.77, 9.60, 8.66, 9.96, 9.13, 10.48, 9.31, 9.59]
L3B_MQM = [8.11, 8.44, 8.03, 7.96, 8.88, 7.70, 7.31, 7.72, 7.63, 7.67, 7.72, 7.23, 7.56, 7.20, 7.55, 7.03, 7.49, 7.19]

total_data = [
    cal_relative(Q1B_bleu, 16.12),
    cal_relative(Q1B_MQM, 4.23, mapping=True),
    cal_relative(Q3B_bleu, 17.64),
    cal_relative(Q3B_MQM, 3.97, mapping=True),
    cal_relative(Q7B_bleu, 16.43),
    cal_relative(Q7B_MQM, 5.10, mapping=True),
]
total_data = {
    "Llama-1B": (cal_relative(L1B_bleu, 2.91), cal_relative(L1B_MQM, 9.65, mapping=True)),
    "Qwen-1.5B": (cal_relative(Q1B_bleu, 16.12), cal_relative(Q1B_MQM, 4.23, mapping=True)),
    "Llama-3B": (cal_relative(L3B_bleu, 6.35), cal_relative(L3B_MQM, 8.22, mapping=True)),
    "Qwen-3B": (cal_relative(Q3B_bleu, 17.64), cal_relative(Q3B_MQM, 3.97, mapping=True))
}


models = total_data.keys()
pd_data = {
    "m": [],
    "k": [],
    "Improvements (%)":[],
    "Metric": [],
    "Model": []
}
for model_name, (bleu, mqm) in total_data.items():
    cnt = 0
    for min_cluster_size in [3,4,5,6]:
        for min_samples in range(1, min_cluster_size+1):
            b_value = bleu[cnt]
            pd_data["m"].append(min_cluster_size)
            pd_data["k"].append(min_samples)
            pd_data["Improvements (%)"].append(b_value)
            pd_data["Metric"].append("BLEU")
            pd_data["Model"].append(model_name)

            m_value = mqm[cnt]
            pd_data["m"].append(min_cluster_size)
            pd_data["k"].append(min_samples)
            pd_data["Improvements (%)"].append(m_value)
            pd_data["Metric"].append("n-MQM")
            pd_data["Model"].append(model_name)
            cnt += 1
            
custom_palette = ["#B85450", "#6C8EBF", "#82B366", "#D6B656"]
g = sns.relplot(
    data=pd_data,
    x="k", y="Improvements (%)", hue="m",
    col="Model", row="Metric",
    kind="line", markers=True,
    dashes=False,
    linewidth = 4, markersize=18, style="m",
    facet_kws={'sharey': False, 'sharex': True},
    height=4, aspect=1.8,
    palette=custom_palette
)
if g._legend:
    # 设置图例标题字体大小
    g._legend.get_title().set_fontsize(33) 
g.tight_layout()
plt.savefig("cluster_hyper.png", dpi=144)
