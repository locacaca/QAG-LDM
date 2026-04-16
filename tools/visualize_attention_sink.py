"""
Attention Sink 可视化脚本

用于分析并可视化记录 attention sink 相关数据：
1. First token attention score (W_sink) 随层变化
2. 最后两层 attention map 热力图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import argparse
from pathlib import Path

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16


def load_data(log_dir: str, gated: bool = True, epoch: int = None):
    """加载日志数据"""
    suffix = "_gated" if gated else "_ungated"
    
    if epoch is None:
        # 找到最新的 epoch
        files = list(Path(log_dir).glob(f"first_token_attn{suffix}_epoch*.json"))
        if not files:
            return None, None
        epochs = [int(f.stem.split("_epoch")[1]) for f in files]
        epoch = max(epochs)
    
    fname1 = os.path.join(log_dir, f"first_token_attn{suffix}_epoch{epoch}.json")
    fname2 = os.path.join(log_dir, f"last_layers_attn{suffix}_epoch{epoch}.json")
    
    with open(fname1, 'r') as f:
        first_token_data = json.load(f)
    
    with open(fname2, 'r') as f:
        last_layers_data = json.load(f)
    
    return first_token_data, last_layers_data


def plot_first_token_attention(log_dir: str, output_dir: str, gated: bool = True, epoch: int = None):
    """
    绘制第一张图: 每层的 first token attention score
    
    X轴: Layer
    Y轴: First Token Attention Score (W_sink)
    """
    first_token_data, _ = load_data(log_dir, gated, epoch)
    if first_token_data is None:
        print(f"No data found in {log_dir}")
        return
    
    data = first_token_data["data"]
    config = first_token_data["config"]
    
    # 按 layer 和 branch 聚合
    layer_text_attn = []
    layer_qual_attn = []
    
    for record in data:
        layer = record["layer"]
        branch = record["branch_name"]
        attn = np.mean(record["first_token_attn"]) if isinstance(record["first_token_attn"], list) else record["first_token_attn"]
        gate = record.get("gate_factor", 1.0)
        
        if branch == "text":
            layer_text_attn.append((layer, attn, gate))
        else:
            layer_qual_attn.append((layer, attn, gate))
    
    # 计算每个 layer 的平均值
    layers = sorted(set(r[0] for r in layer_text_attn + layer_qual_attn))
    
    text_means = []
    text_stds = []
    qual_means = []
    qual_stds = []
    
    for layer in layers:
        text_vals = [r[1] for r in layer_text_attn if r[0] == layer]
        qual_vals = [r[1] for r in layer_qual_attn if r[0] == layer]
        
        text_means.append(np.mean(text_vals) if text_vals else 0)
        text_stds.append(np.std(text_vals) if text_vals else 0)
        qual_means.append(np.mean(qual_vals) if qual_vals else 0)
        qual_stds.append(np.std(qual_vals) if qual_vals else 0)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, text_means, width, label='Text Branch', 
                   yerr=text_stds, capsize=3, alpha=0.8)
    bars2 = ax.bar(x + width/2, qual_means, width, label='Quality Branch',
                   yerr=qual_stds, capsize=3, alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('First Token Attention Score (W_sink)')
    mode_str = "Gated" if gated else "Ungated"
    ax.set_title(f'First Token Attention Score by Layer ({mode_str})')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    suffix = "_gated" if gated else "_ungated"
    output_path = os.path.join(output_dir, f"first_token_attention{suffix}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_attention_heatmap(log_dir: str, output_dir: str, gated: bool = True, epoch: int = None):
    """
    绘制第二张图: 最后两层 attention map 热力图
    
    X轴: Key position (context tokens)
    Y轴: Query position (audio tokens, focus on first few)
    颜色: Normalized attention score
    """
    _, last_layers_data = load_data(log_dir, gated, epoch)
    if last_layers_data is None:
        print(f"No data found in {log_dir}")
        return
    
    data = last_layers_data["data"]
    config = last_layers_data["config"]
    
    # 找到最后两层的数据
    last_two_layers = sorted(set(r["layer"] for r in data))[-2:]
    
    if len(last_two_layers) < 2:
        print("Not enough layers for heatmap")
        return
    
    # 创建 2x2 子图 (text + quality for each layer)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, layer in enumerate(last_two_layers):
        layer_data = [r for r in data if r["layer"] == layer]
        
        for j, branch in enumerate(["text", "quality"]):
            branch_data = [r for r in layer_data if r["branch_name"] == branch]
            
            if not branch_data:
                continue
            
            # 取平均
            attn_maps = [np.array(r["attn_map"]) for r in branch_data]
            attn_mean = np.mean(attn_maps, axis=0)
            
            # 只显示前 20 个 query 位置
            num_queries = min(20, attn_mean.shape[0])
            
            ax = axes[i, j]
            
            # 绘制热力图
            im = ax.imshow(attn_mean[:num_queries, :], aspect='auto', cmap='viridis')
            
            ax.set_xlabel('Key Position (Context)')
            ax.set_ylabel('Query Position (Audio)')
            branch_str = "Text" if branch == "text" else "Quality"
            ax.set_title(f'Layer {layer} - {branch_str} Branch')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Attention Score')
    
    mode_str = "Gated" if gated else "Ungated"
    fig.suptitle(f'Attention Maps - Last Two Layers ({mode_str})', fontsize=18)
    
    plt.tight_layout()
    
    suffix = "_gated" if gated else "_ungated"
    output_path = os.path.join(output_dir, f"attention_heatmap{suffix}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def compare_gated_vs_ungated(log_dir: str, output_dir: str, epoch: int = None):
    """
    对比门控前后的 attention 分布
    """
    gated_data, _ = load_data(log_dir, True, epoch)
    ungated_data, _ = load_data(log_dir, False, epoch)
    
    if gated_data is None or ungated_data is None:
        print("Need both gated and ungated data for comparison")
        return
    
    gated_records = gated_data["data"]
    ungated_records = ungated_data["data"]
    
    # 按 layer 聚合
    layers = sorted(set(r["layer"] for r in gated_records))
    
    gated_means = []
    ungated_means = []
    
    for layer in layers:
        gated_layer = [r for r in gated_records if r["layer"] == layer]
        ungated_layer = [r for r in ungated_records if r["layer"] == layer]
        
        gated_means.append(np.mean([np.mean(r["first_token_attn"]) for r in gated_layer]) if gated_layer else 0)
        ungated_means.append(np.mean([np.mean(r["first_token_attn"]) for r in ungated_layer]) if ungated_layer else 0)
    
    # 绘图对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, gated_means, width, label='Gated', alpha=0.8)
    ax.bar(x + width/2, ungated_means, width, label='Ungated', alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('First Token Attention Score (W_sink)')
    ax.set_title('Comparison: Gated vs Ungated')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "comparison_gated_vs_ungated.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Attention Sink data')
    parser.add_argument('--log_dir', type=str, default='./logs/attention_sink',
                        help='Directory containing log files')
    parser.add_argument('--output_dir', type=str, default='./logs/attention_sink/plots',
                        help='Output directory for plots')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch to visualize (default: latest)')
    parser.add_argument('--mode', type=str, choices=['gated', 'ungated', 'both', 'compare'], 
                        default='both', help='Which mode to visualize')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ['gated', 'both']:
        plot_first_token_attention(args.log_dir, args.output_dir, True, args.epoch)
        plot_attention_heatmap(args.log_dir, args.output_dir, True, args.epoch)
    
    if args.mode in ['ungated', 'both']:
        plot_first_token_attention(args.log_dir, args.output_dir, False, args.epoch)
        plot_attention_heatmap(args.log_dir, args.output_dir, False, args.epoch)
    
    if args.mode == 'compare':
        compare_gated_vs_ungated(args.log_dir, args.output_dir, args.epoch)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
