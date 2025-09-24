import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogFormatter
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')
cluster = [
    [41.61, 44.28, 39.54, 30.90, 31.27, 47.27],
    [37.66, 45.59, 41.72, 30.42, 32.99, 49.32],
    [40.57, 43.39, 38.44, 31.48, 32.49, 47.16],
    [41.33, 44.51, 40.34, 30.02, 32.88, 47.56]
]

entropy = [
    [273.97, 477.85, 729.24, 238.51, 456.55, 664.15],
    [281.24, 475.81, 684.24, 239.97, 448.70, 598.17],
    [270.59, 491.20, 697.36, 235.08, 448.29, 672.89],
    [259.39, 474.33, 664.79, 235.08, 446.61, 611.99]
]

self = [
    [241.82, 395.22, 667.35, 345.31, 834.10, 887.65],
    [233.07, 396.81, 669.50, 322.20, 790.63, 779.19],
    [218.47, 416.12, 676.49, 332.65, 849.52, 986.25],
    [207.64, 413.45, 660.79, 325.56, 781.37, 792.84]
]

for i in range(len(entropy)):
    for j in range(len(entropy[i])):
        entropy[i][j] *= 8
        self[i][j] *= 8




# Qwen_data = {
#     "Scale": ["1.5B"] * 3 + ["3B"] * 3 + ["7B"] * 3,
#     "Method": ["SV", "EM", "LJ"] * 3,
#     "Seconds": [39.47, 2449.92, 2087.12, 45.19, 3938.56, 3670.14, 44.24, 5643.92, 5332.32]
# }

# Llama_data = {
#     "Scale": ["1B"] * 3 + ["3B"] * 3 + ["8B"] * 3,
#     "Method": ["SV", "EM", "LJ"] * 3,
#     "Seconds": [31.43, 1914.16, 2809.28, 34.70, 3662.48, 6614.69, 48.92, 6146.08, 8774.00]
# }
# total_data = {
#     "Models": ["Qwen"] * 9 + ["Llama"] * 9,
#     "Scale": Qwen_data["Scale"] + Llama_data["Scale"],
#     "Method": Qwen_data["Method"] + Llama_data["Method"],
#     "Seconds": Qwen_data["Seconds"] + Llama_data["Seconds"]
# }

sns.set_theme(style="whitegrid", font="Times New Roman")
inner_palette = ["#FAD9D5", "#B0E3E6", "#D0CEE2"]
edge_palette = ["#AE4132", "#0E8088", "#56517E"]
hatches = ["x", "/", "."]
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

qwen_data = {
    "Scale": [],
    "Method": [],
    "Seconds": []
}
scales = ["1.5B", "3B", "7B"]
for i in range(4):
    for j in range(3):
        qwen_data["Scale"].append(scales[j])
        qwen_data["Method"].append("SVSI")
        qwen_data["Seconds"].append(cluster[i][j])
        qwen_data["Scale"].append(scales[j])
        qwen_data["Method"].append("EM")
        qwen_data["Seconds"].append(entropy[i][j])
        qwen_data["Scale"].append(scales[j])
        qwen_data["Method"].append("SJ")
        qwen_data["Seconds"].append(self[i][j])
# g = sns.lineplot(
#     data=qwen_data,
#     x="Scale", y="Seconds", hue="Method", 
#     palette=custom_palette, alpha=1.0, markers=True, style="Method",
#     linewidth = 4, markersize=18, dashes=False,
#     err_style='band',
#     ax=axes[0]
# )
g = sns.barplot(
    data=qwen_data,
    x="Scale", y="Seconds", hue="Method", 
    palette=inner_palette, 
    # edgecolor=sns.color_palette(edge_palette),
    alpha=1.0, linewidth=2,
    width=0.5, errorbar="sd",
    capsize=0.15,
    err_kws={'linewidth': 2},
    ax=axes[0]
)
g.set(yscale="log")
# axes[0].yaxis.set_major_formatter(LogFormatter())
axes[0].set_xlabel("Qwen Models", size=30)
axes[0].set_ylabel("Seconds", size=30)
axes[0].tick_params(axis='both', which='major', labelsize=23)
cnt = 0
for bar in g.patches:
    if cnt >= 9:
        continue
    bar.set_edgecolor(edge_palette[cnt//3])
    bar.set_hatch(hatches[cnt//3])
    cnt+=1

    
llama_data = {
    "Scale": [],
    "Method": [],
    "Seconds": []
}
scales = ["1B", "3B", "8B"]
for i in range(4):
    for j in range(3,6):
        llama_data["Scale"].append(scales[j-3])
        llama_data["Method"].append("SVSI")
        llama_data["Seconds"].append(cluster[i][j])
        llama_data["Scale"].append(scales[j-3])
        llama_data["Method"].append("EM")
        llama_data["Seconds"].append(entropy[i][j])
        llama_data["Scale"].append(scales[j-3])
        llama_data["Method"].append("SJ")
        llama_data["Seconds"].append(self[i][j])
g = sns.barplot(
    data=llama_data,
    x="Scale", y="Seconds", hue="Method", 
    palette=inner_palette, 
    # edgecolor=sns.color_palette(edge_palette), 
    alpha=1.0, linewidth=2,
    width=0.5, errorbar="sd",
    capsize=0.15,
    err_kws={'linewidth': 2},
    ax=axes[1]
)
g.set(yscale="log")
# axes[0].yaxis.set_major_formatter(LogFormatter())
axes[1].set_xlabel("Llama Models", size=30)
axes[1].set_ylabel("Seconds", size=30)
axes[1].tick_params(axis='both', which='major', labelsize=23)
cnt = 0
for bar in g.patches:
    if cnt >= 9:
        continue
    bar.set_edgecolor(edge_palette[cnt//3])
    bar.set_hatch(hatches[cnt//3])
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
axes[1].get_legend().remove()
fig.legend(
    new_handles, 
    labels, 
    frameon=False,
    markerscale=3,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.04),
    ncol=3,
    fontsize=20,
    title_fontsize=20
)
# legend = axes[0].get_legend()
# plt.setp(legend.get_texts(), fontsize=20) 
# plt.setp(legend.get_title(), fontsize=23)     

plt.tight_layout()
plt.savefig("time_compare.png", dpi=144, bbox_inches='tight')
