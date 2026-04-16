import json
import os

def load_fad_summary(file_path):
    """加载FAD摘要文件并返回其内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def print_fad_statistics(summary, quality_score):
    """打印FAD统计信息"""
    if summary and "statistics_by_quality" in summary:
        stats = summary["statistics_by_quality"].get(quality_score)
        if stats:
            total_tasks = summary.get("total_tasks", 0)
            successful = summary.get("successful", 0)
            failed = summary.get("failed", 0)
            count = stats.get("count", 0)
            
            print(f"质量分数 {quality_score}:")
            print(f"  总任务数: {total_tasks}")
            print(f"  成功数: {successful}")
            print(f"  失败数: {failed}")
            print(f"  统计样本数: {count}")
            print(f"  FAD均值: {stats['mean']:.6f}")
            print(f"  FAD最小值: {stats['min']:.6f}")
            print(f"  FAD最大值: {stats['max']:.6f}")
            print(f"  FAD标准差: {stats['std']:.6f}")
            return stats['mean']
        else:
            print(f"没有找到质量分数 {quality_score} 的统计信息")
            return None
    else:
        print("摘要格式错误或没有统计信息")
        return None

def main():
    """主函数"""
    # 文件路径
    file_paths = {
        "0.2": "results/fad_summary_0.2.json",
        "0.5": "results/fad_summary_0.5.json",
        "0.8": "results/fad_summary_0.8.json"
    }
    
    means = {}
    sample_counts = {}
    print("各质量分数FAD统计信息:")
    print("=" * 50)
    
    # 加载和打印每个文件的统计信息
    for quality, path in file_paths.items():
        summary = load_fad_summary(path)
        if summary:
            mean = print_fad_statistics(summary, quality)
            if mean is not None:
                means[quality] = mean
                # 保存样本数量
                if "statistics_by_quality" in summary and quality in summary["statistics_by_quality"]:
                    sample_counts[quality] = summary["statistics_by_quality"][quality].get("count", 0)
        print("-" * 50)
    
    # 计算和打印总体均值
    if means:
        # 计算加权均值 (如果样本数量不同)
        if len(set(sample_counts.values())) > 1:
            total_samples = sum(sample_counts.values())
            weighted_mean = sum(means[q] * sample_counts[q] for q in means) / total_samples
            print(f"所有质量分数的FAD加权均值: {weighted_mean:.6f} (基于不同样本数量)")
            
        # 计算简单均值
        simple_mean = sum(means.values()) / len(means)
        print(f"所有质量分数的FAD简单均值: {simple_mean:.6f}")
        
        print("\n各质量分数FAD均值汇总:")
        for quality, mean in means.items():
            print(f"质量分数 {quality}: {mean:.6f} (样本数: {sample_counts.get(quality, '未知')})")
    else:
        print("没有可用的FAD均值数据")

if __name__ == "__main__":
    main()
