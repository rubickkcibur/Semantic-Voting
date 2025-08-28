import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogFormatter
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')

Qwen_data = {
    "Scale": ["1.5B"] * 3 + ["3B"] * 3 + ["7B"] * 3,
    "Method": ["SV", "EM", "LJ"] * 3,
    "Seconds": [39.47, 2449.92, 2087.12, 45.19, 3938.56, 3670.14, 44.24, 5643.92, 5332.32]
}

Llama_data = {
    "Scale": ["1B"] * 3 + ["3B"] * 3 + ["8B"] * 3,
    "Method": ["SV", "EM", "LJ"] * 3,
    "Seconds": [31.43, 1914.16, 2809.28, 34.70, 3662.48, 6614.69, 48.92, 6146.08, 8774.00]
}
total_data = {
    "Models": ["Qwen"] * 9 + ["Llama"] * 9,
    "Scale": Qwen_data["Scale"] + Llama_data["Scale"],
    "Method": Qwen_data["Method"] + Llama_data["Method"],
    "Seconds": Qwen_data["Seconds"] + Llama_data["Seconds"]
}

sns.set_theme(style="whitegrid", font="Times New Roman")
custom_palette = ["#B85450", "#6C8EBF", "#82B366"]
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

g = sns.barplot(
    data=Llama_data,
    x="Scale", y="Seconds", hue="Method", 
    palette=custom_palette, alpha=.8, 
    width=0.8,
    ax=axes[0]
)
g.set(yscale="log")
axes[0].yaxis.set_major_formatter(LogFormatter())
axes[0].set_xlabel("Llama Models", size=16)
axes[0].set_ylabel("Seconds", size=16)
axes[0].tick_params(axis='both', which='major', labelsize=13)
# plt.despine(left=True)
# g.set_axis_labels("Llama Models", "Seconds", size=16)
# axes[0].tick_params(axis='both', which='major', labelsize=16)
# g.legend.set_title("Methods")
# plt.setp(g.legend.get_texts(), fontsize=12)
g = sns.barplot(
    data=Qwen_data,
    x="Scale", y="Seconds", hue="Method", 
    palette=custom_palette, alpha=.8, 
    width=0.8,
    ax=axes[1]
)
g.set(yscale="log")
axes[1].yaxis.set_major_formatter(LogFormatter())
axes[1].set_xlabel("Qwen Models", size=16)
axes[1].set_ylabel("Seconds", size=16)
axes[1].tick_params(axis='both', which='major', labelsize=13)

plt.savefig("time_compare.png", dpi=300)
