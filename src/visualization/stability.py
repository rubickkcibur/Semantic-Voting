import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')


#              wmt24pp_de, wmt24pp_fr, wmt24pp_ru, wmt24pp_es, cnn, pubmed
# Qwen-1.5B
# Llama-1B
# Qwen-3B
# Llama-3B
# Qwen-7B
# Llama-8B
def map_MQM(num):
    return 1/(1+num)
base = [
    [16.12, 4.23, 13.63, 6.29, 10.63, 4.91, 20.11, 4.96, 13.50, 29.62, None, None],
    [2.91, 9.65, 4.29, 9.99, 1.71, 11.09, 5.47, 10.30, 16.86, 37.93, 27.62, 50.23],
    [17.64, 3.97, 19.69, 4.31, 11.90, 4.86, 23.71, 4.23, None, None, None, None],
    [6.35, 8.22, 8.20, 8.56, 3.63, 9.34, 10.03, 8.83, 7.23, 16.25, 27.77, 49.91],
    [16.42, 5.10, 19.71, 5.00, 11.62, 5.53, 23.67, 5.13, 16.44, 33.90, 18.81, 32.44],
    [20.22, 3.97, 24.32, 3.95, 13.94, 4.50, 28.92, 3.89, 20.68, 45.48, 29.48, 53.68]
]

self = [
    [3.45, 9.92, 6.73, 9.05, 4.06, 9.69, 1.94, 12.39, 13.54, 27.46, None, None],
    [2.06, 10.00, 2.81, 11.01, 1.98, 10.58, 2.57, 11.86, 16.53, 38.02, 28.27, 52.54],
    [18.62, 3.67, 20.02, 4.20, 12.52, 4.53, 23.15, 4.34, None, None, None, None],
    [4.78, 9.12, 4.67, 10.00, 1.26, 11.58, 5.09, 10.82, 0.18, 1.92, 23.01, 43.96],
    [18.34, 4.42, 21.20, 4.40, 12.08, 5.12, 25.58, 4.65, 7.49, 16.33, 23.91, 38.95],
    [20.83, 3.85, 11.46, 7.14, 7.19, 7.32, 26.08, 4.52, 0.08, 1.73, 10.34, 21.08]
]

entropy = [
    [18.21, 3.95, 19.77, 4.13, 11.67, 4.42, 24.32, 3.96, 1.63, 4.41, None, None],
    [1.51, 10.52, 4.35, 10.37, 4.86, 8.00, 5.21, 10.56, 14.45, 25.66, 28.14, 45.69],
    [19.32, 3.87, 19.72, 4.66, 13.66, 4.69, 25.22, 4.09, None, None, None, None],
    [10.52, 6.61, 14.00, 6.12, 11.34, 5.13, 12.01, 7.46, 20.69, 38.97, 29.35, 46.82],
    [7.32, 8.95, 17.66, 6.05, 11.63, 5.69, 17.01, 7.71, 17.65, 33.75, 30.98, 46.10],
    [0, 11.55, 22.35, 4.01, 8.98, 6.78, 12.40, 10.10, 23.23, 44.78, 33.86, 51.62]
]

cluster = [
    [18.04, 3.87, 20.34, 4.01, 11.93, 4.41, 23.46, 4.03, 13.72, 30.63, None, None],
    [5.28, 8.43, 5.59, 9.27, 2.75, 9.99, 6.89, 9.70, 17.83, 40.28, 27.70, 50.92],
    [20.44, 3.66, 20.79, 4.13, 13.41, 4.33, 25.27, 3.99, None, None, None, None],
    [8.71, 7.63, 10.51, 7.62, 5.42, 8.50, 11.89, 8.26, 9.18, 20.36, 27.67, 50.09],
    [20.19, 4.23, 21.76, 4.63, 12.35, 5.14, 25.53, 4.63, 20.69, 42.61, 29.63, 47.76],
    [20.76, 3.78, 24.74, 3.84, 14.54, 4.23, 29.33, 3.89, 20.72, 45.53, 29.69, 54.03]
]


for i in range(len(base)):
    for j in range(len(base[0])):
        if base[i][j] is None:
            self[i][j] = None
            entropy[i][j] = None
            cluster[i][j] = None
        elif (j == 1 or j == 3 or j == 5 or j == 7):
            self[i][j] = (map_MQM(self[i][j]) - map_MQM(base[i][j]))/map_MQM(base[i][j]) * 100
            entropy[i][j] = (map_MQM(entropy[i][j]) - map_MQM(base[i][j]))/map_MQM(base[i][j]) * 100
            cluster[i][j] = (map_MQM(cluster[i][j]) - map_MQM(base[i][j]))/map_MQM(base[i][j]) * 100
        else:
            self[i][j] = (self[i][j] - base[i][j])/base[i][j] * 100
            entropy[i][j] = (entropy[i][j] - base[i][j])/base[i][j] * 100
            cluster[i][j] = (cluster[i][j] - base[i][j])/base[i][j] * 100
dataset = {
    "Method": [],
    "lexical": [],
    "semantic": [],
    "size": []
}

for i in range(len(base)):
    for j in range(len(base[0])//2):
        if base[i][j*2] is None:
            continue
        dataset["Method"].append("SVSI")
        dataset["lexical"].append(cluster[i][j*2])
        dataset["semantic"].append(cluster[i][j*2+1])
        dataset["size"].append(cluster[i][j*2] + cluster[i][j*2+1])

        dataset["Method"].append("EM")
        dataset["lexical"].append(entropy[i][j*2])
        dataset["semantic"].append(entropy[i][j*2+1])
        dataset["size"].append(entropy[i][j*2] + entropy[i][j*2+1])

        dataset["Method"].append("SJ")
        dataset["lexical"].append(self[i][j*2])
        dataset["semantic"].append(self[i][j*2+1])
        dataset["size"].append(self[i][j*2] + self[i][j*2+1])

sns.set_theme(style="white", font = "Times New Roman")
custom_palette = ["#B85450", "#6C8EBF", "#82B366"]

# g = sns.relplot(x="lexical", y="semantic", hue="Method", size="size", 
#             sizes = (40,600), alpha=.7, palette=custom_palette,
#             height=6, data=dataset, aspect=1.5)
# ax = g.ax if hasattr(g, 'ax') else g.axes.flat[0]

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(
    data=dataset,
    x="lexical", 
    y="semantic", 
    hue="Method",
    style="Method",
    markers=["o", "^", "D"],
    size="size",
    sizes=(40, 600),
    alpha=0.7,
    palette=custom_palette,
    ax=ax
)

ax.set_xscale('symlog', base=2)
ax.set_yscale('symlog', base=2)
ax.set_xlim(-126, 600)
ax.set_ylim(-126, 600)

ax.tick_params(axis='both', which='major', labelsize=18)
# g.set_axis_labels("Lexical improvement (%)", "Semantic improvement (%)", size=26)
ax.set_xlabel("Lexical improvement (%)", fontsize=26)
ax.set_ylabel("Semantic improvement (%)", fontsize=26)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
try:
    size_idx = labels.index("size")
except ValueError:
    size_idx = len(labels)  # 如果没有 "size"，就取全部

new_handles = handles[1:size_idx]  # 跳过第一个 "Method" 标题
new_labels = labels[1:size_idx]
ax.legend().remove()
fig.legend(
    new_handles, 
    new_labels, 
    frameon=False,
    markerscale=3,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.04),
    ncol=3,
    fontsize=20,
    title_fontsize=20
)

plt.tight_layout()
plt.savefig("stability_plot.png", dpi=144, bbox_inches='tight')

# leg = g.legend
# handles = leg.legend_handles  # 所有图例图形元素（圆点等）
# labels = [t.get_text() for t in leg.get_texts()]  # 所有图例文字
# # 3. 找到 "size" 标题的位置
# size_idx = labels.index("size")
# new_handles = handles[1:size_idx]  
# new_labels = labels[1:size_idx]
# g.legend.remove()
# g.figure.legend(
#     new_handles, 
#     new_labels, 
#     frameon=False,
#     markerscale=2.0,
#     loc='upper center',        # 图例位置：顶部居中
#     bbox_to_anchor=(0.45, 1.04), # 相对于整个图的位置，1.05 在图上方一点
#     ncol=3,          # 横向排列，每个标签一列
#     fontsize=20,               # 字体大小
#     title_fontsize=20          # 标题字体大小
# )

# g.tight_layout()
# plt.setp(g.legend.get_texts(), fontsize=20)
# plt.savefig("stability_plot.png", dpi=144, bbox_inches='tight')