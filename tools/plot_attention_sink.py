#!/usr/bin/env python
"""Attention Sink Analysis: Baseline vs Gated Attention"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 数据模拟 (替换为你实际的模型层数数据)
# ============================================================
num_layers = 24

# Baseline (Softmax) - 第一个token累积大量注意力
baseline_sink_weights = np.array([
    85.2, 82.1, 79.8, 77.3, 74.5, 71.2,
    68.9, 65.4, 62.1, 58.7, 55.3, 51.8,
    48.2, 44.5, 40.1, 36.7, 32.3, 28.9,
    25.4, 21.8, 18.2, 15.6, 12.1, 8.7
])

# Gated Attention (Ours) - 注意力分布更均匀
gated_sink_weights = np.array([
    12.3, 11.8, 11.2, 10.5, 9.8, 9.1,
    8.4, 7.7, 7.1, 6.5, 5.9, 5.3,
    4.8, 4.2, 3.7, 3.2, 2.8, 2.4,
    2.1, 1.8, 1.5, 1.3, 1.1, 0.9
])

# ============================================================
# 绘图配置
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# 横轴：层数 (1-24)
layers = np.arange(1, num_layers + 1)

# 绘制 Baseline
ax.plot(layers, baseline_sink_weights,
        color='#E63946',           # 红色
        linestyle='--',             # 虚线
        marker='o',                 # 圆点
        markersize=6,
        linewidth=2,
        label='Baseline (Softmax)',
        markerfacecolor='white',
        markeredgewidth=2)

# 绘制 Gated Attention
ax.plot(layers, gated_sink_weights,
        color='#1D3557',            # 深蓝色
        linestyle='-',             # 实线
        marker='s',                 # 方块
        markersize=6,
        linewidth=2,
        label='Ours (Gated-HS)',
        markerfacecolor='white',
        markeredgewidth=2)

# ============================================================
# 参考线与区域
# ============================================================
ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(y=10, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)

# 添加50%参考线标签
ax.text(num_layers + 0.3, 50, '50%', va='center', fontsize=10, color='gray')

# ============================================================
# 坐标轴设置
# ============================================================
ax.set_xlim(0.5, num_layers + 0.5)
ax.set_ylim(0, 100)
ax.set_xlabel('Layer Index', fontweight='normal')
ax.set_ylabel('Avg. Attention Weight on First Token (%)', fontweight='normal')

ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
ax.set_xticklabels(['1', '4', '8', '12', '16', '20', '24'])

# ============================================================
# 标注均值差异
# ============================================================
baseline_mean = np.mean(baseline_sink_weights)
gated_mean = np.mean(gated_sink_weights)

# 文字标注框
textstr = (f'Baseline (Softmax) Mean: {baseline_mean:.1f}%\n'
           f'Ours (Gated-HS) Mean: {gated_mean:.1f}%\n'
           f'Reduction: {baseline_mean - gated_mean:.1f}%')

props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.97, textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props,
        fontfamily='serif')

# ============================================================
# 图例与标题
# ============================================================
ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

ax.set_title('Attention Sink Analysis: Baseline vs Gated Attention',
             fontweight='normal', pad=15)

# 网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# ============================================================
# 保存与显示
# ============================================================
plt.tight_layout()
plt.savefig('attention_sink_analysis.pdf', dpi=300, bbox_inches='tight',
            format='pdf', facecolor='white')
plt.savefig('attention_sink_analysis.png', dpi=300, bbox_inches='tight',
            format='png', facecolor='white')

print(f"Figure saved as 'attention_sink_analysis.pdf'")
print(f"Baseline Mean: {baseline_mean:.1f}%")
print(f"Gated Mean: {gated_mean:.1f}%")
print(f"Reduction: {baseline_mean - gated_mean:.1f}%")

plt.show()
