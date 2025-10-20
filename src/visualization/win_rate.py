import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogFormatter
fm.fontManager.addfont('/mnt/maclabcv2/rubickjiang/codes/fonts/times.ttf')

SV_base = {
    "Llama-1B": [442, 356],
    "Llama-3B": [417, 386],
    "Llama-8B": [400, 400],
    "Qwen-1.5B": [400, 400],
    "Qwen-3B": [413, 387],
    "Qwen-7B": [400, 400],
}

Models = ["Llama-1B", "Llama-3B", "Llama-8B", "Qwen-1.5B", "Qwen-3B", "Qwen-7B"]
Models = Models[::-1]
SV_win = [SV_base[model][0] / (SV_base[model][0] + SV_base[model][1])*100 for model in Models]
base_win = [SV_base[model][1] / (SV_base[model][0] + SV_base[model][1])*100 for model in Models]

sns.set_theme(style="white", font="Times New Roman")
inner_palette = ["#FAD9D5", "#B0E3E6"]
edge_palette = ["#AE4132", "#0E8088"]
hatchs = ["x", "/"]
fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.barh(Models, SV_win, 
            color=inner_palette[0], 
            label='SVSI wins', 
            edgecolor=edge_palette[0], 
            linewidth=0,
            height=0.4,
            hatch=hatchs[0]
        )
bars2 = ax.barh(Models, base_win, 
            left=SV_win,
            color=inner_palette[1], 
            label='Base wins', 
            edgecolor=edge_palette[1], 
            linewidth=0,
            height=0.4,
            hatch=hatchs[1]
        )

for i, (m1, m2) in enumerate(zip(SV_win, base_win)):
    if m1 > 0.05:  # 只有当值足够大时才显示标签
        ax.text(m1/2, i, f'{m1:.1f}', ha='center', va='center', fontweight='bold', fontsize=23)
    if m2 > 0.05:
        ax.text(m1 + m2/2, i, f'{m2:.1f}', ha='center', va='center', fontweight='bold', fontsize=23)
ax.set_xlabel('Win Rate (%)', fontsize=25)
ax.set_yticklabels(Models, fontsize=23)
ax.set_xticks([])
ax.legend(
    loc='upper center',
    ncol=2,
    fontsize=23,
    bbox_to_anchor=(0.5, 1.1),
    frameon=False,
    markerscale=3,
)
# ax.set_xlim(0, 1.1)  # 确保显示完整

# 美化图表
# sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('win_rate.png', dpi=144, bbox_inches='tight')
