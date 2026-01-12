#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果分析与可视化脚本

功能：
1. 收集所有实验结果
2. 计算每种方法的平均值和标准差
3. 生成带误差棒的对比图
4. 生成LaTeX格式的表格
5. 生成Markdown格式的表格

Usage:
    python analyze_results.py                    # 分析所有结果
    python analyze_results.py --output_dir results_analysis  # 指定输出目录
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# 方法配置（用于排序和显示）
# =============================================================================

METHOD_ORDER = ["CE", "FocalLoss", "CB", "LDAM", "CRT", "LOS", "LTSEI"]

METHOD_DISPLAY_NAMES = {
    "CE": "CE (Baseline)",
    "FocalLoss": "Focal Loss",
    "CB": "CB Loss",
    "LDAM": "LDAM",
    "CRT": "CRT",
    "LOS": "LOS",
    "LTSEI": "LTSEI (Ours)",
}

METHOD_COLORS = {
    "CE": "#7f7f7f",         # 灰色
    "FocalLoss": "#ff7f0e",  # 橙色
    "CB": "#2ca02c",         # 绿色
    "LDAM": "#d62728",       # 红色
    "CRT": "#9467bd",        # 紫色
    "LOS": "#17becf",        # 青色
    "LTSEI": "#1f77b4",      # 蓝色（我们的方法）
}


# =============================================================================
# 指标定义
# =============================================================================

METRICS = {
    # 主要指标
    "accuracy": ("Overall Accuracy", "OA", "%"),
    "balanced_accuracy": ("Balanced Accuracy", "bAcc", "%"),
    "macro_f1": ("Macro F1", "F1", "%"),
    
    # 分组指标
    "majority_acc": ("Majority Acc", "Many", "%"),
    "medium_acc": ("Medium Acc", "Med", "%"),
    "minority_acc": ("Minority Acc", "Few", "%"),
}


# =============================================================================
# 数据收集函数
# =============================================================================

def collect_results_from_experiments(workspace: str = ".") -> Dict[str, List[Dict]]:
    """
    从experiments目录收集所有实验结果
    
    Returns:
        Dict[method_name, List[result_dict]]
    """
    experiments_dir = Path(workspace) / "experiments"
    if not experiments_dir.exists():
        print(f"警告: experiments目录不存在: {experiments_dir}")
        return {}
    
    results_by_method = defaultdict(list)
    
    # 遍历所有实验目录
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # 解析方法名和种子
        dir_name = exp_dir.name
        method_name = None
        seed = None
        
        for method in METHOD_ORDER:
            if dir_name.startswith(f"{method}_seed"):
                method_name = method
                try:
                    # 格式: METHOD_seedN_TIMESTAMP
                    seed_part = dir_name.split("_seed")[1].split("_")[0]
                    seed = int(seed_part)
                except:
                    continue
                break
        
        if method_name is None:
            # 尝试匹配LTSEI的其他可能名称
            if "moe" in dir_name.lower() or "ltsei" in dir_name.lower():
                method_name = "LTSEI"
                try:
                    if "_seed" in dir_name:
                        seed_part = dir_name.split("_seed")[1].split("_")[0]
                        seed = int(seed_part)
                    else:
                        seed = 0  # 默认种子
                except:
                    seed = 0
        
        if method_name is None:
            continue
        
        # 加载结果
        results_file = exp_dir / "results" / "results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 提取关键指标
            test_results = result_data.get("test_results", {})
            overall = test_results.get("overall", {})
            group_wise = test_results.get("group_wise", {})
            
            extracted = {
                "method": method_name,
                "seed": seed,
                "exp_dir": str(exp_dir),
                # 主要指标
                "accuracy": overall.get("accuracy", 0),
                "balanced_accuracy": overall.get("balanced_accuracy", 0),
                "macro_f1": overall.get("macro_f1", 0),
                # 分组指标
                "majority_acc": group_wise.get("majority", {}).get("accuracy", 0),
                "medium_acc": group_wise.get("medium", {}).get("accuracy", 0),
                "minority_acc": group_wise.get("minority", {}).get("accuracy", 0),
                # 时间信息
                "total_time_ms": result_data.get("timing", {}).get("total_ms", 0),
            }
            
            results_by_method[method_name].append(extracted)
            
        except Exception as e:
            print(f"警告: 无法加载结果 {results_file}: {e}")
    
    return dict(results_by_method)


def collect_results_from_batch_file(batch_file: str) -> Dict[str, List[Dict]]:
    """
    从批量结果文件收集结果
    """
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_results = json.load(f)
    
    results_by_method = defaultdict(list)
    
    for item in batch_results:
        if item.get("status") != "success":
            continue
        
        method_name = item.get("method")
        seed = item.get("seed")
        result_data = item.get("results", {})
        
        if not result_data:
            continue
        
        test_results = result_data.get("test_results", {})
        overall = test_results.get("overall", {})
        group_wise = test_results.get("group_wise", {})
        
        extracted = {
            "method": method_name,
            "seed": seed,
            "exp_dir": item.get("exp_dir", ""),
            "accuracy": overall.get("accuracy", 0),
            "balanced_accuracy": overall.get("balanced_accuracy", 0),
            "macro_f1": overall.get("macro_f1", 0),
            "majority_acc": group_wise.get("majority", {}).get("accuracy", 0),
            "medium_acc": group_wise.get("medium", {}).get("accuracy", 0),
            "minority_acc": group_wise.get("minority", {}).get("accuracy", 0),
            "total_time_ms": result_data.get("timing", {}).get("total_ms", 0),
        }
        
        results_by_method[method_name].append(extracted)
    
    return dict(results_by_method)


# =============================================================================
# 统计分析函数
# =============================================================================

def compute_statistics(results_by_method: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Dict]]:
    """
    计算每种方法的统计量（均值、标准差、最大、最小）
    
    Returns:
        Dict[method, Dict[metric, Dict[stat_name, value]]]
    """
    stats = {}
    
    for method, results in results_by_method.items():
        if not results:
            continue
        
        method_stats = {}
        
        for metric_key in METRICS.keys():
            values = [r.get(metric_key, 0) for r in results]
            values = [v for v in values if v is not None and v > 0]
            
            if not values:
                method_stats[metric_key] = {
                    "mean": 0,
                    "std": 0,
                    "min": 0,
                    "max": 0,
                    "count": 0,
                }
            else:
                method_stats[metric_key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }
        
        stats[method] = method_stats
    
    return stats


# =============================================================================
# 可视化函数
# =============================================================================

def plot_bar_with_error(
    stats: Dict[str, Dict[str, Dict]],
    metrics: List[str],
    output_path: str,
    title: str = "Method Comparison",
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
):
    """
    绘制带误差棒的柱状图
    """
    # 过滤有结果的方法，并按顺序排列
    methods = [m for m in METHOD_ORDER if m in stats]
    
    if not methods:
        print("没有可用的结果数据")
        return
    
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    x = np.arange(n_methods)
    width = 0.8 / n_metrics
    
    for i, metric in enumerate(metrics):
        means = [stats[m].get(metric, {}).get("mean", 0) for m in methods]
        stds = [stats[m].get(metric, {}).get("std", 0) for m in methods]
        
        offset = (i - n_metrics / 2 + 0.5) * width
        
        metric_info = METRICS.get(metric, (metric, metric, "%"))
        bars = ax.bar(
            x + offset, means, width,
            label=metric_info[1],
            yerr=stds,
            capsize=3,
            error_kw={'linewidth': 1},
        )
    
    # 设置标签
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], rotation=15, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 设置y轴范围
    all_means = []
    for metric in metrics:
        all_means.extend([stats[m].get(metric, {}).get("mean", 0) for m in methods])
    
    if all_means:
        y_min = max(0, min(all_means) - 10)
        y_max = min(100, max(all_means) + 10)
        ax.set_ylim(y_min, y_max)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存: {output_path}")


def plot_grouped_comparison(
    stats: Dict[str, Dict[str, Dict]],
    output_path: str,
    dpi: int = 300,
):
    """
    绘制分组对比图（Many vs Medium vs Few）
    """
    methods = [m for m in METHOD_ORDER if m in stats]
    
    if not methods:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    
    group_metrics = [
        ("majority_acc", "Many-shot Classes"),
        ("medium_acc", "Medium-shot Classes"),
        ("minority_acc", "Few-shot Classes"),
    ]
    
    for ax, (metric, title) in zip(axes, group_metrics):
        means = [stats[m].get(metric, {}).get("mean", 0) for m in methods]
        stds = [stats[m].get(metric, {}).get("std", 0) for m in methods]
        colors = [METHOD_COLORS.get(m, "#333333") for m in methods]
        
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, 
                     error_kw={'linewidth': 1.5})
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                          rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 设置y轴范围
        if means:
            y_min = max(0, min(means) - 15)
            y_max = min(100, max(means) + 10)
            ax.set_ylim(y_min, y_max)
        
        # 在柱子上显示数值
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"分组对比图已保存: {output_path}")


def plot_radar_chart(
    stats: Dict[str, Dict[str, Dict]],
    output_path: str,
    dpi: int = 300,
):
    """
    绘制雷达图（多维度对比）
    """
    methods = [m for m in METHOD_ORDER if m in stats]
    
    if not methods:
        return
    
    metrics_to_show = ["accuracy", "balanced_accuracy", "majority_acc", "medium_acc", "minority_acc"]
    metric_labels = ["OA", "bAcc", "Many", "Med", "Few"]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_show), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi, subplot_kw=dict(projection='polar'))
    
    for method in methods:
        values = [stats[method].get(m, {}).get("mean", 0) for m in metrics_to_show]
        values += values[:1]  # 闭合
        
        color = METHOD_COLORS.get(method, "#333333")
        ax.plot(angles, values, 'o-', linewidth=2, label=METHOD_DISPLAY_NAMES.get(method, method),
               color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Method Performance Radar", fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"雷达图已保存: {output_path}")


def plot_main_comparison(
    stats: Dict[str, Dict[str, Dict]],
    output_path: str,
    dpi: int = 300,
):
    """
    绘制主要指标对比图（单图多指标）
    """
    methods = [m for m in METHOD_ORDER if m in stats]
    
    if not methods:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    metrics_to_show = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_labels = ["Overall Acc", "Balanced Acc", "Macro F1"]
    hatches = ['', '///', '...']
    
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (metric, label, hatch) in enumerate(zip(metrics_to_show, metric_labels, hatches)):
        means = [stats[m].get(metric, {}).get("mean", 0) for m in methods]
        stds = [stats[m].get(metric, {}).get("std", 0) for m in methods]
        
        bars = ax.bar(x + (i - 1) * width, means, width, 
                     label=label, yerr=stds, capsize=3,
                     hatch=hatch, edgecolor='black', linewidth=0.5,
                     error_kw={'linewidth': 1})
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Performance Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods], 
                      rotation=15, ha='right', fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 设置y轴范围
    all_values = []
    for metric in metrics_to_show:
        all_values.extend([stats[m].get(metric, {}).get("mean", 0) for m in methods])
    
    if all_values:
        y_min = max(0, min(all_values) - 10)
        y_max = min(100, max(all_values) + 10)
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"主对比图已保存: {output_path}")


# =============================================================================
# 表格生成函数
# =============================================================================

def generate_latex_table(
    stats: Dict[str, Dict[str, Dict]],
    output_path: str,
    metrics: Optional[List[str]] = None,
):
    """
    生成LaTeX格式的表格
    """
    if metrics is None:
        metrics = ["accuracy", "balanced_accuracy", "macro_f1", "majority_acc", "minority_acc"]
    
    methods = [m for m in METHOD_ORDER if m in stats]
    
    if not methods:
        return
    
    # 找出每个指标的最佳方法
    best_methods = {}
    for metric in metrics:
        best_val = 0
        best_method = None
        for m in methods:
            val = stats[m].get(metric, {}).get("mean", 0)
            if val > best_val:
                best_val = val
                best_method = m
        best_methods[metric] = best_method
    
    # 生成LaTeX表格
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Performance comparison of different methods}")
    lines.append("\\label{tab:method_comparison}")
    
    # 列格式
    col_format = "l" + "c" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")
    
    # 表头
    header = "Method"
    for metric in metrics:
        info = METRICS.get(metric, (metric, metric, "%"))
        header += f" & {info[1]}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # 数据行
    for method in methods:
        row = METHOD_DISPLAY_NAMES.get(method, method)
        
        for metric in metrics:
            mean = stats[method].get(metric, {}).get("mean", 0)
            std = stats[method].get(metric, {}).get("std", 0)
            
            if best_methods[metric] == method:
                row += f" & \\textbf{{{mean:.2f}}}$\\pm${std:.2f}"
            else:
                row += f" & {mean:.2f}$\\pm${std:.2f}"
        
        row += " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX表格已保存: {output_path}")


def generate_markdown_table(
    stats: Dict[str, Dict[str, Dict]],
    output_path: str,
    metrics: Optional[List[str]] = None,
):
    """
    生成Markdown格式的表格
    """
    if metrics is None:
        metrics = ["accuracy", "balanced_accuracy", "macro_f1", "majority_acc", "minority_acc"]
    
    methods = [m for m in METHOD_ORDER if m in stats]
    
    if not methods:
        return
    
    # 找出每个指标的最佳方法
    best_methods = {}
    for metric in metrics:
        best_val = 0
        best_method = None
        for m in methods:
            val = stats[m].get(metric, {}).get("mean", 0)
            if val > best_val:
                best_val = val
                best_method = m
        best_methods[metric] = best_method
    
    lines = []
    lines.append("# 方法性能对比表\n")
    
    # 表头
    header = "| Method |"
    separator = "|:------|"
    
    for metric in metrics:
        info = METRICS.get(metric, (metric, metric, "%"))
        header += f" {info[1]} |"
        separator += ":---:|"
    
    lines.append(header)
    lines.append(separator)
    
    # 数据行
    for method in methods:
        row = f"| {METHOD_DISPLAY_NAMES.get(method, method)} |"
        
        for metric in metrics:
            mean = stats[method].get(metric, {}).get("mean", 0)
            std = stats[method].get(metric, {}).get("std", 0)
            
            if best_methods[metric] == method:
                row += f" **{mean:.2f}±{std:.2f}** |"
            else:
                row += f" {mean:.2f}±{std:.2f} |"
        
        lines.append(row)
    
    # 添加注释
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**指标说明:**")
    for metric in metrics:
        info = METRICS.get(metric, (metric, metric, "%"))
        lines.append(f"- **{info[1]}**: {info[0]}")
    
    lines.append("")
    lines.append(f"*结果基于 {stats[methods[0]].get('accuracy', {}).get('count', '?')} 个随机种子的平均值和标准差*")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Markdown表格已保存: {output_path}")


def generate_summary_text(
    stats: Dict[str, Dict[str, Dict]],
    output_path: str,
):
    """
    生成文本格式的摘要报告
    """
    methods = [m for m in METHOD_ORDER if m in stats]
    
    if not methods:
        return
    
    lines = []
    lines.append("=" * 80)
    lines.append("实验结果摘要报告")
    lines.append("=" * 80)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"方法数量: {len(methods)}")
    lines.append("")
    
    # 每种方法的详细结果
    for method in methods:
        lines.append("-" * 60)
        lines.append(f"【{METHOD_DISPLAY_NAMES.get(method, method)}】")
        lines.append("-" * 60)
        
        for metric, info in METRICS.items():
            s = stats[method].get(metric, {})
            if s.get("count", 0) > 0:
                lines.append(f"  {info[0]:25s}: {s['mean']:6.2f}% ± {s['std']:5.2f}% "
                           f"(min: {s['min']:.2f}, max: {s['max']:.2f}, n={s['count']})")
        
        lines.append("")
    
    # 最佳方法总结
    lines.append("=" * 80)
    lines.append("最佳方法总结")
    lines.append("=" * 80)
    
    for metric, info in METRICS.items():
        best_val = 0
        best_method = None
        for m in methods:
            val = stats[m].get(metric, {}).get("mean", 0)
            if val > best_val:
                best_val = val
                best_method = m
        
        if best_method:
            lines.append(f"  {info[0]:25s}: {METHOD_DISPLAY_NAMES.get(best_method, best_method)} "
                       f"({best_val:.2f}%)")
    
    lines.append("")
    lines.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"摘要报告已保存: {output_path}")
    
    # 同时打印到控制台
    print('\n'.join(lines))


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="分析实验结果并生成图表")
    parser.add_argument("--batch_file", type=str, default=None,
                       help="批量结果JSON文件路径（可选）")
    parser.add_argument("--output_dir", type=str, default="results_analysis",
                       help="输出目录")
    parser.add_argument("--dpi", type=int, default=300,
                       help="图片DPI")
    
    args = parser.parse_args()
    
    workspace = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(workspace) / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # 收集结果
    print("\n收集实验结果...")
    
    if args.batch_file:
        results_by_method = collect_results_from_batch_file(args.batch_file)
    else:
        results_by_method = collect_results_from_experiments(workspace)
    
    if not results_by_method:
        print("未找到任何实验结果！")
        print("请先运行实验: python run_all_experiments.py")
        return
    
    # 打印找到的结果
    print(f"\n找到 {len(results_by_method)} 种方法的结果:")
    for method, results in results_by_method.items():
        print(f"  - {method}: {len(results)} 个种子")
    
    # 计算统计量
    print("\n计算统计量...")
    stats = compute_statistics(results_by_method)
    
    # 保存统计量
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计量已保存: {stats_file}")
    
    # 生成图表
    print("\n生成图表...")
    
    # 1. 主对比图
    plot_main_comparison(
        stats,
        str(output_dir / "comparison_main.png"),
        dpi=args.dpi
    )
    
    # 2. 分组对比图
    plot_grouped_comparison(
        stats,
        str(output_dir / "comparison_groups.png"),
        dpi=args.dpi
    )
    
    # 3. 雷达图
    plot_radar_chart(
        stats,
        str(output_dir / "radar_chart.png"),
        dpi=args.dpi
    )
    
    # 4. 带误差棒的柱状图
    plot_bar_with_error(
        stats,
        ["accuracy", "balanced_accuracy", "macro_f1"],
        str(output_dir / "bar_with_error.png"),
        title="Performance Comparison with Standard Deviation",
        dpi=args.dpi
    )
    
    # 生成表格
    print("\n生成表格...")
    
    generate_latex_table(
        stats,
        str(output_dir / "table.tex")
    )
    
    generate_markdown_table(
        stats,
        str(output_dir / "table.md")
    )
    
    generate_summary_text(
        stats,
        str(output_dir / "summary.txt")
    )
    
    print(f"\n分析完成！所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
