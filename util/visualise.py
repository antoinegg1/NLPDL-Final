import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 -----------------------------------------------------------------
metrics = ["Style Transfer Strength", "Content Preservation", "Fluency"]

# paragraph数据（模型名称: [STS, CP, Fluency]）
paragraph_scores = {
    "Mistral": [0.7234793491864832, 0.8937421777221529, 0.8588986232790992],
    "Qwen": [0.6180602006688962, 0.7856187290969899, 0.7110367892976589],
    "GPT2-finetuned": [0.24415519399249064, 0.19118898623279104, 0.32615769712140175],
    "GPT2-pretrained": [0.2622895622895623, 0.22356902356902358, 0.3727272727272727],
    "GPT2": [0.2461279461279461, 0.20875420875420875, 0.3501683501683502],
}

# sentence数据（模型名称: [STS, CP, Fluency]）
sentence_scores = {
    "Mistral": [0.7329999999999999, 0.8970000000000001, 0.8736666666666667],
    "Qwen": [0.5504504504504505, 0.7384384384384385, 0.6790790790790792],
    "GPT2-finetuned": [0.22100685733400235, 0.19541729386184983, 0.33191169091821365],
    "GPT2-pretrained": [0.24040404040404043, 0.21346801346801347, 0.39124579124579123],
    "GPT2": [0.26107382550335567, 0.2389261744966443, 0.4003355704697987],
}

# 2. 配色（参考示例截图） -------------------------------------------------------
# 你可以根据实际需求修改颜色值以更贴近示例图
color_map = {
    "Mistral": "#9B59B6",        # 紫色
    "Qwen": "#00008B",           # 深蓝
    "GPT2-finetuned": "#4169E1", # 亮蓝
    "GPT2-pretrained": "#7FB3D5",# 浅蓝
    "GPT2": "#2ECC71",           # 亮绿色
}

# 3. 创建子图并绘制 ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 通用设置
bar_width = 0.12

# ------------------ 绘制 paragraph 子图 ------------------
ax_paragraph = axes[0]
# ax_paragraph.set_title("Paragraph Results")
x = np.arange(len(metrics))
offsets = np.linspace(
    - (len(paragraph_scores) - 1) * bar_width / 2,
    (len(paragraph_scores) - 1) * bar_width / 2,
    len(paragraph_scores)
)

for i, (model_name, scores) in enumerate(paragraph_scores.items()):
    ax_paragraph.bar(
        x + offsets[i],
        scores,
        width=bar_width,
        label=model_name,
        color=color_map.get(model_name, "#333333")  # 若未指定，则默认灰色
    )

ax_paragraph.set_xticks(x)
ax_paragraph.set_xticklabels(metrics, rotation=15)
ax_paragraph.set_ylabel("Scores")
ax_paragraph.grid(axis="y", linestyle="--", alpha=0.7)

# ------------------ 绘制 sentence 子图 ------------------
ax_sentence = axes[1]
x2 = np.arange(len(metrics))
offsets2 = np.linspace(
    - (len(sentence_scores) - 1) * bar_width / 2,
    (len(sentence_scores) - 1) * bar_width / 2,
    len(sentence_scores)
)

for i, (model_name, scores) in enumerate(sentence_scores.items()):
    ax_sentence.bar(
        x2 + offsets2[i],
        scores,
        width=bar_width,
        label=model_name,
        color=color_map.get(model_name, "#333333")
    )

ax_sentence.set_xticks(x2)
ax_sentence.set_xticklabels(metrics, rotation=15)
ax_sentence.grid(axis="y", linestyle="--", alpha=0.7)

# 4. 图例 & 布局 & 保存 -------------------------------------------------------
# 只需要在其中一个子图上加图例即可，也可分开加
handles, labels = ax_paragraph.get_legend_handles_labels()
fig.legend(handles, labels, title="Model", loc="upper right", bbox_to_anchor=(1, 1),)

plt.tight_layout()

# 将结果保存为 PDF
plt.savefig("style_transfer_results.pdf", format="pdf")

# 如果你同时也想在屏幕显示，取消下行注释
# plt.show()
