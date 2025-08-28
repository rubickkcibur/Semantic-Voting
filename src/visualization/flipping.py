import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')

def map_MQM(num):
    return (15-num)/15

base = [
    [16.12, 4.23, 13.63, 6.29, 10.63, 4.91, 20.11, 4.96, 13.50, 29.62, None, None],
    [2.91, 9.65, 4.29, 9.99, 1.71, 11.09, 5.47, 10.30, 16.86, 37.93, 27.62, 50.23],
    [17.64, 3.97, 19.69, 4.31, 11.90, 4.86, 23.71, 4.23, None, None, None, None],
    [6.35, 8.22, 8.20, 8.56, 3.63, 9.34, 10.03, 8.83, 7.23, 16.25, 27.77, 49.91],
    [16.42, 5.10, 19.71, 5.00, 11.62, 5.53, 23.67, 5.13, 16.44, 33.90, 18.81, 32.44],
    [20.22, 3.97, 24.32, 3.95, 13.94, 4.50, 28.92, 3.89, 20.68, 45.48, 29.48, 53.68]
]

dpo = [
    [18.04, 3.87, 20.34, 4.01, 11.93, 4.41, 23.46, 4.03, 13.72, 30.63, None, None],
    [8.18, 7.20, 5.59, 9.27, 7.23, 7.04, 6.89, 9.70, 17.83, 40.28, 27.70, 50.92],
    [20.44, 3.66, 20.79, 4.13, 13.41, 4.33, 25.27, 3.99, None, None, None, None],
    [8.52, 7.50, 11.13, 7.36, 5.04, 8.50, 12.08, 8.17, 9.18, 20.36, 27.67, 50.09],
    [20.19, 4.23, 21.76, 4.63, 12.35, 5.14, 25.53, 4.63, 20.69, 42.61, 29.63, 47.76],
    [20.76, 3.78, 24.74, 3.84, 14.54, 4.23, 29.33, 3.89, 20.72, 45.53, 29.69, 54.03]
]

flipping_dpo = [
    [3.46, 9.60, 3.86, 10.43, 3.40, 10.70, 4.06, 12.02, 13.02, 27.66, None, None],
    [0.27, 11.15, 0.49, 11.84, 0.14, 13.02, 0.57, 12.94, 17.24, 40.00, 27.39, 49.84],
    [14.11, 5.04, 16.98, 4.96, 10.44, 5.45, 20.65, 4.82, None, None, None, None],
    [4.27, 9.23, 2.96, 10.75, 2.29, 10.44, 5.39, 10.30, 5.68, 13.35, 27.31, 48.87],
    [15.43, 5.26, 18.64, 4.92, 12.36, 5.14, 22.94, 5.22, 16.03, 32.25, 20.57, 36.31],
    [7.81, 7.83, 5.37, 9.55, 3.07, 9.72, 11.08, 7.93, 20.27, 44.92, 8.44, 18.27]
]

for i in range(len(base)):
    for j in range(len(base[0])):
        if base[i][j] is None:
            dpo[i][j] = 0
            flipping_dpo[i][j] = 0
        elif (j == 1 or j == 3 or j == 5 or j == 7):
            dpo[i][j] = (map_MQM(dpo[i][j]) - map_MQM(base[i][j]))/base[i][j] * 100
            flipping_dpo[i][j] = (map_MQM(flipping_dpo[i][j]) - map_MQM(base[i][j]))/base[i][j] * 100
        else:
            dpo[i][j] = (dpo[i][j] - base[i][j])/base[i][j] * 100
            flipping_dpo[i][j] = (flipping_dpo[i][j] - base[i][j])/base[i][j] * 100
        base[i][j] = 0


sns.set_theme(style="whitegrid", font="Times New Roman")
fig, axes = plt.subplots(2, 6, figsize=(28, 12), subplot_kw=dict(polar=True), )

data_length = 6
angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
labels = ["Qwen-1.5B", "Llama-1B", "Qwen-3B", "Llama-3B", "Qwen-7B", "Llama-8B"]
labels = labels + [labels[0]]
tasks_map = ["wmt24pp_de", "wmt24pp_fr", "wmt24pp_ru", "wmt24pp_es", "cnn_dailymail", "pubmed"]
metric_map_MT = ["BLEU", "MQM"]
metric_map_TS = ["RougeL", "BLEURT"]
for subplot_row in range(2):
    for subplot_column in range(6):
        ax = axes[subplot_row, subplot_column]
        data_j = subplot_column * 2 + subplot_row
        base_column = [row[data_j] for row in base]
        base_column = np.concatenate((base_column, [base_column[0]]))
        dpo_column = [row[data_j] for row in dpo]
        dpo_column = np.concatenate((dpo_column, [dpo_column[0]]))
        flipping_dpo_column = [row[data_j] for row in flipping_dpo]
        flipping_dpo_column = np.concatenate((flipping_dpo_column, [flipping_dpo_column[0]]))

        ax.plot(angles, dpo_column, color='#B85450')
        ax.fill(angles, dpo_column, color='#F8CECC', alpha=0.6)
        ax.plot(angles, base_column, color='#6C8EBF')
        ax.fill(angles, base_column, color='#DAE8FC', alpha=0.6)
        ax.plot(angles, flipping_dpo_column, color='#D6B656')
        ax.fill(angles, flipping_dpo_column, color='#FFF2CC', alpha=0.6)

        ax.set_thetagrids(angles*180/np.pi, labels, fontsize=22)
        ax.set_theta_zero_location('N')
        # ax.set_rlim(0, 100)
        ax.set_rlabel_position(180) #刻度朝向
        label = "{} ({})".format(
            tasks_map[subplot_column],
            metric_map_MT[subplot_row] if subplot_column < 4 else metric_map_TS[subplot_row]
        )
        # ax.set_title(label)
# plt.legend(["base", "dpo", "flipping_dpo"], loc='best')
handles = [
    plt.Line2D([0], [0], color='#6C8EBF', lw=2, label='base'),
    plt.Line2D([0], [0], color='#B85450', lw=2, label='dpo'),
    plt.Line2D([0], [0], color='#D6B656', lw=2, label='flipping_dpo')
]
fig.legend(handles=handles, loc='center', ncol=3, fontsize = 25)
plt.tight_layout()
plt.savefig("flipping.png", dpi=300)