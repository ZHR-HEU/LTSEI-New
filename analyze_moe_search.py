#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoE Search Results Analysis and Visualization
=============================================

分析MoE超参数搜索结果，生成详细报告和可视化图表。

功能：
1. 参数敏感性分析
2. 交互效应分析
3. 收敛性分析
4. 统计显著性检验
5. 可视化图表生成

Author: Auto-generated
Date: 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# 数据加载
# =============================================================================

def load_search_results(search_dir: str) -> Tuple[List[Dict], Dict]:
    """加载搜索结果"""
    results = []
    summary = {}

    # 尝试加载all_results.json
    all_results_path = os.path.join(search_dir, "all_results.json")
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            results = json.load(f)

    # 尝试加载final_summary.json
    summary_path = os.path.join(search_dir, "final_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)

    return results, summary


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """将结果转换为DataFrame"""
    rows = []

    for r in results:
        if r.get("status") != "success" or r.get("metrics") is None:
            continue

        row = {
            "seed": r["seed"],
            "exp_name": r.get("exp_name", ""),
            "status": r["status"],
            "duration": r.get("duration", 0),
        }

        # 添加参数
        for k, v in r.get("params", {}).items():
            row[f"param_{k}"] = v

        # 添加指标
        for k, v in r.get("metrics", {}).items():
            row[f"metric_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 统计分析
# =============================================================================

def compute_parameter_sensitivity(df: pd.DataFrame, metric: str = "metric_balanced_accuracy") -> Dict:
    """计算参数敏感性"""
    param_cols = [c for c in df.columns if c.startswith("param_")]
    sensitivity = {}

    for param in param_cols:
        param_name = param.replace("param_", "")
        unique_values = df[param].unique()

        if len(unique_values) < 2:
            continue

        # 按参数值分组计算统计量
        groups = df.groupby(param)[metric].agg(['mean', 'std', 'count']).reset_index()
        groups.columns = ['value', 'mean', 'std', 'count']

        # 计算方差分析 (ANOVA)
        group_values = [df[df[param] == v][metric].values for v in unique_values]
        group_values = [g for g in group_values if len(g) > 0]

        if len(group_values) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*group_values)
            except:
                f_stat, p_value = np.nan, np.nan
        else:
            f_stat, p_value = np.nan, np.nan

        # 计算效应量 (eta-squared)
        if not np.isnan(f_stat):
            ss_between = sum(len(g) * (np.mean(g) - df[metric].mean())**2 for g in group_values)
            ss_total = sum((df[metric] - df[metric].mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
        else:
            eta_squared = np.nan

        # 最佳值
        best_idx = groups['mean'].idxmax()
        best_value = groups.loc[best_idx, 'value']
        best_mean = groups.loc[best_idx, 'mean']

        sensitivity[param_name] = {
            "groups": groups.to_dict('records'),
            "f_statistic": float(f_stat) if not np.isnan(f_stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "eta_squared": float(eta_squared) if not np.isnan(eta_squared) else None,
            "best_value": best_value,
            "best_mean": float(best_mean),
            "range": float(groups['mean'].max() - groups['mean'].min()),
        }

    # 按效应量排序
    sensitivity = dict(sorted(
        sensitivity.items(),
        key=lambda x: x[1].get('eta_squared', 0) if x[1].get('eta_squared') is not None else 0,
        reverse=True
    ))

    return sensitivity


def compute_interaction_effects(df: pd.DataFrame, metric: str = "metric_balanced_accuracy") -> Dict:
    """计算参数交互效应"""
    param_cols = [c for c in df.columns if c.startswith("param_")]
    interactions = {}

    for i, param1 in enumerate(param_cols):
        for param2 in param_cols[i+1:]:
            name1 = param1.replace("param_", "")
            name2 = param2.replace("param_", "")

            # 创建交互项
            interaction_means = df.groupby([param1, param2])[metric].mean().reset_index()
            interaction_means.columns = [name1, name2, 'mean']

            # 计算最佳组合
            best_idx = interaction_means['mean'].idxmax()
            best_combo = interaction_means.iloc[best_idx]

            interactions[f"{name1}_x_{name2}"] = {
                "param1": name1,
                "param2": name2,
                "best_combo": {
                    name1: best_combo[name1],
                    name2: best_combo[name2],
                    "mean": float(best_combo['mean']),
                },
                "data": interaction_means.to_dict('records'),
            }

    return interactions


def perform_statistical_tests(df: pd.DataFrame, metric: str = "metric_balanced_accuracy") -> Dict:
    """执行统计显著性检验"""
    tests = {}

    # 按参数配置分组
    param_cols = [c for c in df.columns if c.startswith("param_")]
    df['config_key'] = df[param_cols].apply(lambda x: tuple(x), axis=1)

    # 获取排名前5的配置
    config_means = df.groupby('config_key')[metric].mean().sort_values(ascending=False)
    top_configs = config_means.head(5).index.tolist()

    if len(top_configs) >= 2:
        # 配对t检验：最佳 vs 次佳
        best_values = df[df['config_key'] == top_configs[0]][metric].values
        second_values = df[df['config_key'] == top_configs[1]][metric].values

        if len(best_values) >= 2 and len(second_values) >= 2:
            t_stat, p_value = stats.ttest_ind(best_values, second_values)
            tests['best_vs_second'] = {
                "test": "independent t-test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_0.05": p_value < 0.05,
                "best_mean": float(np.mean(best_values)),
                "second_mean": float(np.mean(second_values)),
            }

    # 与默认配置对比
    default_key = tuple([0.1, 1.0, 1.0, 30.0, 0.25, 0.5])  # 默认参数
    default_mask = df['config_key'] == default_key

    if default_mask.sum() > 0 and len(top_configs) > 0:
        default_values = df[default_mask][metric].values
        best_values = df[df['config_key'] == top_configs[0]][metric].values

        if len(default_values) >= 2 and len(best_values) >= 2:
            t_stat, p_value = stats.ttest_ind(best_values, default_values)
            tests['best_vs_default'] = {
                "test": "independent t-test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_0.05": p_value < 0.05,
                "improvement": float(np.mean(best_values) - np.mean(default_values)),
            }

    return tests


# =============================================================================
# 可视化
# =============================================================================

def plot_parameter_sensitivity(df: pd.DataFrame, sensitivity: Dict, output_dir: str):
    """绘制参数敏感性图"""
    if not HAS_MATPLOTLIB:
        return

    n_params = len(sensitivity)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (param_name, data) in enumerate(sensitivity.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        groups = pd.DataFrame(data['groups'])

        if HAS_SEABORN:
            sns.barplot(data=groups, x='value', y='mean', ax=ax, palette='viridis')
        else:
            ax.bar(range(len(groups)), groups['mean'])
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups['value'])

        # 添加误差棒
        ax.errorbar(range(len(groups)), groups['mean'], yerr=groups['std'],
                   fmt='none', color='black', capsize=3)

        ax.set_xlabel(param_name)
        ax.set_ylabel('Balanced Accuracy (%)')
        ax.set_title(f'{param_name}\n(η²={data["eta_squared"]:.3f}, p={data["p_value"]:.3e})'
                    if data["p_value"] is not None else param_name)

        # 高亮最佳值
        best_idx = groups[groups['value'] == data['best_value']].index[0]
        ax.patches[best_idx].set_facecolor('red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_interaction_heatmaps(df: pd.DataFrame, interactions: Dict, output_dir: str):
    """绘制参数交互效应热力图"""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    # 选择最重要的交互
    key_interactions = [
        'w_balance_x_gate_tau',
        'scale_x_ldam_power',
        'ldam_power_x_ldam_max_m',
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, key in enumerate(key_interactions):
        if key not in interactions:
            continue

        ax = axes[idx]
        data = interactions[key]

        # 创建透视表
        pivot_df = pd.DataFrame(data['data'])
        pivot_table = pivot_df.pivot(
            index=data['param1'],
            columns=data['param2'],
            values='mean'
        )

        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'{data["param1"]} × {data["param2"]}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interaction_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_configs_comparison(df: pd.DataFrame, output_dir: str, top_n: int = 10):
    """绘制Top配置对比图"""
    if not HAS_MATPLOTLIB:
        return

    param_cols = [c for c in df.columns if c.startswith("param_")]
    df['config_key'] = df[param_cols].apply(lambda x: tuple(x), axis=1)

    # 聚合
    config_stats = df.groupby('config_key')['metric_balanced_accuracy'].agg(['mean', 'std', 'count'])
    config_stats = config_stats.sort_values('mean', ascending=False).head(top_n)
    config_stats = config_stats.reset_index()

    # 创建标签
    labels = []
    for config in config_stats['config_key']:
        param_names = [c.replace('param_', '') for c in param_cols]
        label = '\n'.join([f"{n[:3]}={v}" for n, v in zip(param_names, config)])
        labels.append(label)

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(range(top_n), config_stats['mean'], xerr=config_stats['std'],
                  capsize=3, color='steelblue', alpha=0.8)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Balanced Accuracy (%)')
    ax.set_title(f'Top {top_n} Configurations')
    ax.invert_yaxis()

    # 添加数值标签
    for i, (mean, std) in enumerate(zip(config_stats['mean'], config_stats['std'])):
        ax.text(mean + std + 0.5, i, f'{mean:.2f}±{std:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_configs_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_seed_variance(df: pd.DataFrame, output_dir: str):
    """绘制种子方差分析图"""
    if not HAS_MATPLOTLIB:
        return

    param_cols = [c for c in df.columns if c.startswith("param_")]
    df['config_key'] = df[param_cols].apply(lambda x: str(tuple(x)), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 种子间方差分布
    ax1 = axes[0]
    config_stds = df.groupby('config_key')['metric_balanced_accuracy'].std()

    if HAS_SEABORN:
        sns.histplot(config_stds, bins=20, ax=ax1, kde=True)
    else:
        ax1.hist(config_stds, bins=20, alpha=0.7)

    ax1.axvline(config_stds.mean(), color='red', linestyle='--', label=f'Mean: {config_stds.mean():.2f}')
    ax1.set_xlabel('Standard Deviation across Seeds')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Seed Variance')
    ax1.legend()

    # 2. 按种子的整体表现
    ax2 = axes[1]
    seed_means = df.groupby('seed')['metric_balanced_accuracy'].mean()

    if HAS_SEABORN:
        sns.barplot(x=seed_means.index, y=seed_means.values, ax=ax2, palette='viridis')
    else:
        ax2.bar(seed_means.index, seed_means.values)

    ax2.set_xlabel('Seed')
    ax2.set_ylabel('Mean Balanced Accuracy (%)')
    ax2.set_title('Performance by Seed')
    ax2.axhline(seed_means.mean(), color='red', linestyle='--', label=f'Overall Mean: {seed_means.mean():.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_variance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_correlations(df: pd.DataFrame, output_dir: str):
    """绘制指标相关性热力图"""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return

    metric_cols = [c for c in df.columns if c.startswith("metric_")]

    if len(metric_cols) < 2:
        return

    corr_matrix = df[metric_cols].corr()

    # 重命名列
    corr_matrix.columns = [c.replace('metric_', '') for c in corr_matrix.columns]
    corr_matrix.index = [c.replace('metric_', '') for c in corr_matrix.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, ax=ax, square=True)
    ax.set_title('Metric Correlations')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 报告生成
# =============================================================================

def generate_detailed_report(
    df: pd.DataFrame,
    sensitivity: Dict,
    interactions: Dict,
    stat_tests: Dict,
    summary: Dict,
    output_dir: str
) -> str:
    """生成详细分析报告"""

    report_path = os.path.join(output_dir, "detailed_analysis_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# MoE Hyperparameter Search - Detailed Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # =====================================================================
        # 执行摘要
        # =====================================================================
        f.write("## Executive Summary\n\n")

        total_exps = len(df)
        success_rate = total_exps / summary.get('total_experiments', total_exps) * 100

        f.write(f"- **Total Experiments**: {summary.get('total_experiments', total_exps)}\n")
        f.write(f"- **Successful Experiments**: {total_exps}\n")
        f.write(f"- **Success Rate**: {success_rate:.1f}%\n")
        f.write(f"- **Unique Configurations**: {df.groupby([c for c in df.columns if c.startswith('param_')]).ngroups}\n")
        f.write(f"- **Seeds per Configuration**: {summary.get('config', {}).get('num_seeds', 10)}\n\n")

        # 最佳配置
        if 'final_best' in summary:
            f.write("### Best Configuration Found\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for k, v in summary['final_best'].items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        # =====================================================================
        # 参数敏感性分析
        # =====================================================================
        f.write("## Parameter Sensitivity Analysis\n\n")

        f.write("Parameters ranked by effect size (η²):\n\n")
        f.write("| Rank | Parameter | Best Value | Best Mean | Range | η² | p-value | Significance |\n")
        f.write("|------|-----------|------------|-----------|-------|-----|---------|-------------|\n")

        for rank, (param, data) in enumerate(sensitivity.items(), 1):
            eta = data.get('eta_squared', 0)
            p = data.get('p_value', 1)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            f.write(f"| {rank} | {param} | {data['best_value']} | {data['best_mean']:.2f}% | ")
            f.write(f"{data['range']:.2f}% | {eta:.4f} | {p:.2e} | {sig} |\n")

        f.write("\n*Significance: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant*\n\n")

        # 详细参数分析
        f.write("### Detailed Parameter Analysis\n\n")

        for param, data in sensitivity.items():
            f.write(f"#### {param}\n\n")

            f.write("| Value | Mean | Std | Count |\n")
            f.write("|-------|------|-----|-------|\n")
            for g in data['groups']:
                f.write(f"| {g['value']} | {g['mean']:.2f}% | {g['std']:.2f} | {g['count']} |\n")
            f.write("\n")

            # 建议
            f.write(f"**Recommendation**: Use `{param}={data['best_value']}` ")
            f.write(f"(achieves {data['best_mean']:.2f}% mean balanced accuracy)\n\n")

        # =====================================================================
        # 交互效应分析
        # =====================================================================
        f.write("## Interaction Effects\n\n")

        f.write("Key parameter interactions and their best combinations:\n\n")

        for name, data in interactions.items():
            f.write(f"### {data['param1']} × {data['param2']}\n\n")
            f.write(f"Best combination: `{data['param1']}={data['best_combo'][data['param1']]}`, ")
            f.write(f"`{data['param2']}={data['best_combo'][data['param2']]}` ")
            f.write(f"→ **{data['best_combo']['mean']:.2f}%**\n\n")

        # =====================================================================
        # 统计检验
        # =====================================================================
        f.write("## Statistical Tests\n\n")

        if 'best_vs_second' in stat_tests:
            test = stat_tests['best_vs_second']
            f.write("### Best vs Second Best Configuration\n\n")
            f.write(f"- **Test**: Independent t-test\n")
            f.write(f"- **Best Mean**: {test['best_mean']:.2f}%\n")
            f.write(f"- **Second Mean**: {test['second_mean']:.2f}%\n")
            f.write(f"- **t-statistic**: {test['t_statistic']:.3f}\n")
            f.write(f"- **p-value**: {test['p_value']:.4f}\n")
            f.write(f"- **Significant at α=0.05**: {'Yes' if test['significant_at_0.05'] else 'No'}\n\n")

        if 'best_vs_default' in stat_tests:
            test = stat_tests['best_vs_default']
            f.write("### Best vs Default Configuration\n\n")
            f.write(f"- **Test**: Independent t-test\n")
            f.write(f"- **Improvement**: {test['improvement']:.2f}%\n")
            f.write(f"- **t-statistic**: {test['t_statistic']:.3f}\n")
            f.write(f"- **p-value**: {test['p_value']:.4f}\n")
            f.write(f"- **Significant at α=0.05**: {'Yes' if test['significant_at_0.05'] else 'No'}\n\n")

        # =====================================================================
        # 种子稳定性
        # =====================================================================
        f.write("## Seed Stability Analysis\n\n")

        param_cols = [c for c in df.columns if c.startswith("param_")]
        df_copy = df.copy()
        df_copy['config_key'] = df_copy[param_cols].apply(lambda x: tuple(x), axis=1)

        config_stds = df_copy.groupby('config_key')['metric_balanced_accuracy'].std()

        f.write(f"- **Mean Std across Seeds**: {config_stds.mean():.2f}%\n")
        f.write(f"- **Max Std**: {config_stds.max():.2f}%\n")
        f.write(f"- **Min Std**: {config_stds.min():.2f}%\n\n")

        # 高方差配置警告
        high_var_configs = config_stds[config_stds > config_stds.mean() + config_stds.std()]
        if len(high_var_configs) > 0:
            f.write(f"**Warning**: {len(high_var_configs)} configurations show high variance across seeds.\n\n")

        # =====================================================================
        # 运行时间分析
        # =====================================================================
        if 'duration' in df.columns:
            f.write("## Runtime Analysis\n\n")

            durations = df['duration'].dropna()
            if len(durations) > 0:
                f.write(f"- **Total Runtime**: {durations.sum()/3600:.2f} hours\n")
                f.write(f"- **Average per Experiment**: {durations.mean()/60:.2f} minutes\n")
                f.write(f"- **Std**: {durations.std()/60:.2f} minutes\n\n")

        # =====================================================================
        # 结论与建议
        # =====================================================================
        f.write("## Conclusions and Recommendations\n\n")

        f.write("### Key Findings\n\n")

        # 按敏感性排序的关键发现
        sorted_params = sorted(sensitivity.items(),
                              key=lambda x: x[1].get('eta_squared', 0) or 0,
                              reverse=True)

        for i, (param, data) in enumerate(sorted_params[:3], 1):
            eta = data.get('eta_squared', 0)
            f.write(f"{i}. **{param}** is the {'most' if i==1 else 'second most' if i==2 else 'third most'} ")
            f.write(f"influential parameter (η²={eta:.4f}), with optimal value **{data['best_value']}**\n")

        f.write("\n### Recommended Configuration\n\n")
        f.write("Based on the analysis, the recommended MoE configuration is:\n\n")
        f.write("```yaml\n")
        f.write("moe_config:\n")
        if 'final_best' in summary:
            for k, v in summary['final_best'].items():
                f.write(f"  {k}: {v}\n")
        f.write("```\n\n")

        f.write("---\n")
        f.write("*This report was automatically generated by the MoE hyperparameter analysis tool.*\n")

    print(f"Detailed report saved to: {report_path}")
    return report_path


# =============================================================================
# 主函数
# =============================================================================

def analyze_search_results(search_dir: str, output_dir: str = None):
    """分析搜索结果"""

    if output_dir is None:
        output_dir = os.path.join(search_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("MoE Hyperparameter Search Analysis")
    print(f"{'='*60}")
    print(f"Search directory: {search_dir}")
    print(f"Output directory: {output_dir}")

    # 加载数据
    print("\nLoading results...")
    results, summary = load_search_results(search_dir)

    if not results:
        print("Error: No results found!")
        return

    print(f"Loaded {len(results)} experiment results")

    # 转换为DataFrame
    df = results_to_dataframe(results)
    print(f"Valid experiments: {len(df)}")

    if len(df) == 0:
        print("Error: No valid results to analyze!")
        return

    # 保存原始数据
    df.to_csv(os.path.join(output_dir, "results_dataframe.csv"), index=False)

    # 统计分析
    print("\nPerforming statistical analysis...")

    sensitivity = compute_parameter_sensitivity(df)
    print(f"  - Parameter sensitivity computed")

    interactions = compute_interaction_effects(df)
    print(f"  - Interaction effects computed")

    stat_tests = perform_statistical_tests(df)
    print(f"  - Statistical tests completed")

    # 保存分析结果
    with open(os.path.join(output_dir, "sensitivity_analysis.json"), 'w') as f:
        json.dump(sensitivity, f, indent=2, default=str)

    with open(os.path.join(output_dir, "interaction_effects.json"), 'w') as f:
        json.dump(interactions, f, indent=2, default=str)

    with open(os.path.join(output_dir, "statistical_tests.json"), 'w') as f:
        json.dump(stat_tests, f, indent=2, default=str)

    # 可视化
    print("\nGenerating visualizations...")

    if HAS_MATPLOTLIB:
        plot_parameter_sensitivity(df, sensitivity, output_dir)
        print("  - Parameter sensitivity plot saved")

        plot_top_configs_comparison(df, output_dir)
        print("  - Top configs comparison plot saved")

        plot_seed_variance(df, output_dir)
        print("  - Seed variance analysis plot saved")

        if HAS_SEABORN:
            plot_interaction_heatmaps(df, interactions, output_dir)
            print("  - Interaction heatmaps saved")

            plot_metric_correlations(df, output_dir)
            print("  - Metric correlations plot saved")
    else:
        print("  - Skipping visualizations (matplotlib not available)")

    # 生成报告
    print("\nGenerating detailed report...")
    report_path = generate_detailed_report(df, sensitivity, interactions, stat_tests, summary, output_dir)

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")

    # 打印关键发现摘要
    print(f"\n{'='*60}")
    print("Key Findings Summary")
    print(f"{'='*60}")

    print("\nParameter Importance (by effect size η²):")
    for i, (param, data) in enumerate(list(sensitivity.items())[:6], 1):
        eta = data.get('eta_squared', 0) or 0
        print(f"  {i}. {param}: η²={eta:.4f}, best={data['best_value']} ({data['best_mean']:.2f}%)")

    if 'final_best' in summary:
        print(f"\nBest Configuration:")
        for k, v in summary['final_best'].items():
            print(f"  {k}: {v}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE hyperparameter search results")
    parser.add_argument("search_dir", type=str, help="Directory containing search results")
    parser.add_argument("--output", type=str, default=None, help="Output directory for analysis")

    args = parser.parse_args()

    analyze_search_results(args.search_dir, args.output)


if __name__ == "__main__":
    main()
