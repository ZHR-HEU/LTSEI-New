#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoE Hyperparameter Search Script
================================

对MoE模块的6个超参数进行系统搜索：
1. w_balance: 负载平衡损失权重
2. gate_tau: 门控网络温度
3. la_tau: LogitAdjusted专家温度
4. scale: 余弦分类器缩放因子
5. ldam_power: LDAM边距幂次
6. ldam_max_m: LDAM最大边距

搜索策略：分组网格搜索
- 组1: 门控参数 (w_balance, gate_tau)
- 组2: LDAM参数 (scale, ldam_power, ldam_max_m)
- 组3: LA参数 (la_tau)

每个配置运行10个种子，使用GPU 3

Author: Auto-generated
Date: 2025
"""

import os
import sys
import json
import time
import itertools
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


# =============================================================================
# 搜索配置
# =============================================================================

# MoE参数搜索空间
MOE_SEARCH_SPACE = {
    # 组1: 门控参数
    "w_balance": [0.0, 0.01, 0.05, 0.1, 0.5],
    "gate_tau": [0.5, 1.0, 1.5, 2.0],

    # 组2: LDAM参数 (Expert 1)
    "scale": [10.0, 20.0, 30.0, 50.0],
    "ldam_power": [0.1, 0.25, 0.5],
    "ldam_max_m": [0.3, 0.5, 0.7],

    # 组3: LA参数 (Expert 2)
    "la_tau": [0.5, 1.0, 1.5, 2.0],
}

# 默认值
DEFAULT_VALUES = {
    "w_balance": 0.1,
    "gate_tau": 1.0,
    "scale": 30.0,
    "ldam_power": 0.25,
    "ldam_max_m": 0.5,
    "la_tau": 1.0,
}

# 实验配置
EXPERIMENT_CONFIG = {
    "num_seeds": 10,
    "base_seeds": list(range(42, 52)),  # [42, 43, ..., 51]
    "gpu_id": "3",
    "base_config": "config.yaml",
    "output_dir": "moe_search_results",
}


# =============================================================================
# 工具函数
# =============================================================================

def create_experiment_name(params: Dict[str, float], seed: int) -> str:
    """生成实验名称"""
    param_str = "_".join([f"{k[:3]}{v}" for k, v in sorted(params.items())])
    return f"moe_s{seed}_{param_str}"


def run_single_experiment(
    params: Dict[str, float],
    seed: int,
    gpu_id: str,
    output_base: str,
    config_path: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """运行单个实验"""

    exp_name = create_experiment_name(params, seed)

    # 构建hydra覆盖参数
    overrides = [
        f"seed={seed}",
        f"gpus={gpu_id}",
        f"exp_name={exp_name}",
        f"stage2.moe_config.w_balance={params['w_balance']}",
        f"stage2.moe_config.gate_tau={params['gate_tau']}",
        f"stage2.moe_config.la_tau={params['la_tau']}",
        f"stage2.moe_config.scale={params['scale']}",
        f"stage2.moe_config.ldam_power={params['ldam_power']}",
        f"stage2.moe_config.ldam_max_m={params['ldam_max_m']}",
        # 禁用可视化加速实验
        "visualization.enabled=false",
        "visualization.plot_tsne_2d=false",
        "visualization.plot_tsne_3d=false",
    ]

    cmd = ["python", "main.py"] + overrides

    result = {
        "exp_name": exp_name,
        "params": params.copy(),
        "seed": seed,
        "cmd": " ".join(cmd),
        "status": "pending",
        "metrics": None,
        "error": None,
        "duration": None,
    }

    if dry_run:
        result["status"] = "dry_run"
        print(f"[DRY RUN] {exp_name}")
        print(f"  Command: {' '.join(cmd)}")
        return result

    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Params: {params}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # 运行实验
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2小时超时
        )

        duration = time.time() - start_time
        result["duration"] = duration

        if process.returncode == 0:
            result["status"] = "success"
            # 尝试读取结果
            result["metrics"] = extract_metrics_from_output(process.stdout, exp_name)
        else:
            result["status"] = "failed"
            result["error"] = process.stderr[-2000:] if process.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Experiment timed out after 2 hours"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def extract_metrics_from_output(stdout: str, exp_name: str) -> Optional[Dict[str, float]]:
    """从输出中提取指标"""
    metrics = {}

    # 尝试从experiments目录读取results.json
    exp_dirs = list(Path("experiments").glob(f"{exp_name}_*"))
    if exp_dirs:
        latest_dir = max(exp_dirs, key=lambda p: p.stat().st_mtime)
        results_file = latest_dir / "results" / "results.json"

        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    test_results = data.get("test_results", {})
                    overall = test_results.get("overall", {})
                    group_wise = test_results.get("group_wise", {})

                    metrics = {
                        "accuracy": overall.get("accuracy", 0),
                        "balanced_accuracy": overall.get("balanced_accuracy", 0),
                        "macro_f1": overall.get("macro_f1", 0),
                        "majority_acc": group_wise.get("majority", {}).get("accuracy", 0),
                        "medium_acc": group_wise.get("medium", {}).get("accuracy", 0),
                        "minority_acc": group_wise.get("minority", {}).get("accuracy", 0),
                    }

                    # 计算Head-Tail调和平均
                    head = metrics["majority_acc"]
                    tail = metrics["minority_acc"]
                    if head + tail > 0:
                        metrics["harmonic_mean"] = 2 * head * tail / (head + tail)
                    else:
                        metrics["harmonic_mean"] = 0

                    return metrics
            except Exception as e:
                print(f"Warning: Failed to read results.json: {e}")

    # 备用：从stdout解析
    lines = stdout.split('\n')
    for line in lines:
        if "准确率:" in line or "Accuracy:" in line:
            try:
                val = float(line.split(":")[-1].strip().replace("%", ""))
                metrics["accuracy"] = val
            except:
                pass
        if "平衡准确率:" in line or "Balanced" in line:
            try:
                val = float(line.split(":")[-1].strip().replace("%", ""))
                metrics["balanced_accuracy"] = val
            except:
                pass

    return metrics if metrics else None


def save_results(results: List[Dict], output_dir: str, filename: str = "search_results.json"):
    """保存搜索结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")
    return output_path


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """聚合多种子结果"""
    # 按参数配置分组
    param_groups = {}

    for r in results:
        if r["status"] != "success" or r["metrics"] is None:
            continue

        # 创建参数键
        param_key = tuple(sorted(r["params"].items()))

        if param_key not in param_groups:
            param_groups[param_key] = {
                "params": r["params"],
                "metrics_list": [],
                "seeds": [],
            }

        param_groups[param_key]["metrics_list"].append(r["metrics"])
        param_groups[param_key]["seeds"].append(r["seed"])

    # 计算统计量
    aggregated = []
    for param_key, group in param_groups.items():
        metrics_list = group["metrics_list"]

        if not metrics_list:
            continue

        # 计算每个指标的均值和标准差
        stats = {}
        metric_names = metrics_list[0].keys()

        for metric in metric_names:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_std"] = np.std(values)
                stats[f"{metric}_min"] = np.min(values)
                stats[f"{metric}_max"] = np.max(values)

        aggregated.append({
            "params": group["params"],
            "num_seeds": len(group["seeds"]),
            "seeds": group["seeds"],
            "stats": stats,
        })

    # 按balanced_accuracy排序
    aggregated.sort(key=lambda x: x["stats"].get("balanced_accuracy_mean", 0), reverse=True)

    return {
        "total_configs": len(aggregated),
        "results": aggregated,
    }


# =============================================================================
# 搜索策略
# =============================================================================

def generate_group1_configs() -> List[Dict[str, float]]:
    """组1: 门控参数搜索 (w_balance, gate_tau)"""
    configs = []
    for w_balance in MOE_SEARCH_SPACE["w_balance"]:
        for gate_tau in MOE_SEARCH_SPACE["gate_tau"]:
            config = DEFAULT_VALUES.copy()
            config["w_balance"] = w_balance
            config["gate_tau"] = gate_tau
            configs.append(config)
    return configs


def generate_group2_configs(best_group1: Dict[str, float]) -> List[Dict[str, float]]:
    """组2: LDAM参数搜索 (scale, ldam_power, ldam_max_m)"""
    configs = []
    for scale in MOE_SEARCH_SPACE["scale"]:
        for ldam_power in MOE_SEARCH_SPACE["ldam_power"]:
            for ldam_max_m in MOE_SEARCH_SPACE["ldam_max_m"]:
                config = best_group1.copy()
                config["scale"] = scale
                config["ldam_power"] = ldam_power
                config["ldam_max_m"] = ldam_max_m
                configs.append(config)
    return configs


def generate_group3_configs(best_group2: Dict[str, float]) -> List[Dict[str, float]]:
    """组3: LA参数搜索 (la_tau)"""
    configs = []
    for la_tau in MOE_SEARCH_SPACE["la_tau"]:
        config = best_group2.copy()
        config["la_tau"] = la_tau
        configs.append(config)
    return configs


def generate_all_configs() -> List[Dict[str, float]]:
    """生成所有配置（完整网格搜索）"""
    configs = []

    keys = list(MOE_SEARCH_SPACE.keys())
    values = [MOE_SEARCH_SPACE[k] for k in keys]

    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


# =============================================================================
# 主搜索流程
# =============================================================================

def run_grouped_search(
    gpu_id: str = "3",
    num_seeds: int = 10,
    output_dir: str = "moe_search_results",
    dry_run: bool = False,
    resume_from: str = None,
):
    """
    分组网格搜索

    搜索顺序：
    1. 组1: 搜索 w_balance × gate_tau (20组合)
    2. 组2: 固定组1最优，搜索 scale × ldam_power × ldam_max_m (36组合)
    3. 组3: 固定组1+2最优，搜索 la_tau (4组合)

    总计: (20 + 36 + 4) × 10 seeds = 600 实验
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = os.path.join(output_dir, f"grouped_search_{timestamp}")
    os.makedirs(search_dir, exist_ok=True)

    seeds = list(range(42, 42 + num_seeds))
    all_results = []

    # 如果有断点续传
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        with open(resume_from, 'r') as f:
            checkpoint = json.load(f)
            all_results = checkpoint.get("results", [])
            completed_keys = {(tuple(sorted(r["params"].items())), r["seed"])
                           for r in all_results if r["status"] == "success"}
    else:
        completed_keys = set()

    print("\n" + "="*80)
    print("MoE Hyperparameter Search - Grouped Grid Search")
    print("="*80)
    print(f"GPU: {gpu_id}")
    print(f"Seeds: {seeds}")
    print(f"Output: {search_dir}")
    print("="*80)

    # =========================================================================
    # 组1: 门控参数
    # =========================================================================
    print("\n" + "="*80)
    print("GROUP 1: Gate Parameters (w_balance × gate_tau)")
    print("="*80)

    group1_configs = generate_group1_configs()
    print(f"Total configs: {len(group1_configs)}")
    print(f"Total experiments: {len(group1_configs) * num_seeds}")

    group1_results = []
    for config in group1_configs:
        for seed in seeds:
            param_key = (tuple(sorted(config.items())), seed)
            if param_key in completed_keys:
                print(f"Skipping (already completed): {config}, seed={seed}")
                continue

            result = run_single_experiment(
                params=config,
                seed=seed,
                gpu_id=gpu_id,
                output_base=search_dir,
                config_path="config.yaml",
                dry_run=dry_run,
            )
            group1_results.append(result)
            all_results.append(result)

            # 保存中间结果
            save_results(all_results, search_dir, "checkpoint.json")

    # 分析组1结果，找最优配置
    group1_agg = aggregate_results(group1_results)
    save_results(group1_agg, search_dir, "group1_aggregated.json")

    if group1_agg["results"]:
        best_group1 = group1_agg["results"][0]["params"]
        print(f"\nBest Group 1 config: {best_group1}")
        print(f"Balanced Acc: {group1_agg['results'][0]['stats'].get('balanced_accuracy_mean', 0):.2f}%")
    else:
        best_group1 = DEFAULT_VALUES.copy()
        print("Warning: No successful Group 1 results, using defaults")

    # =========================================================================
    # 组2: LDAM参数
    # =========================================================================
    print("\n" + "="*80)
    print("GROUP 2: LDAM Parameters (scale × ldam_power × ldam_max_m)")
    print("="*80)

    group2_configs = generate_group2_configs(best_group1)
    print(f"Total configs: {len(group2_configs)}")
    print(f"Total experiments: {len(group2_configs) * num_seeds}")

    group2_results = []
    for config in group2_configs:
        for seed in seeds:
            param_key = (tuple(sorted(config.items())), seed)
            if param_key in completed_keys:
                print(f"Skipping (already completed): {config}, seed={seed}")
                continue

            result = run_single_experiment(
                params=config,
                seed=seed,
                gpu_id=gpu_id,
                output_base=search_dir,
                config_path="config.yaml",
                dry_run=dry_run,
            )
            group2_results.append(result)
            all_results.append(result)

            save_results(all_results, search_dir, "checkpoint.json")

    group2_agg = aggregate_results(group2_results)
    save_results(group2_agg, search_dir, "group2_aggregated.json")

    if group2_agg["results"]:
        best_group2 = group2_agg["results"][0]["params"]
        print(f"\nBest Group 2 config: {best_group2}")
        print(f"Balanced Acc: {group2_agg['results'][0]['stats'].get('balanced_accuracy_mean', 0):.2f}%")
    else:
        best_group2 = best_group1.copy()
        print("Warning: No successful Group 2 results, using Group 1 best")

    # =========================================================================
    # 组3: LA参数
    # =========================================================================
    print("\n" + "="*80)
    print("GROUP 3: LogitAdjusted Parameters (la_tau)")
    print("="*80)

    group3_configs = generate_group3_configs(best_group2)
    print(f"Total configs: {len(group3_configs)}")
    print(f"Total experiments: {len(group3_configs) * num_seeds}")

    group3_results = []
    for config in group3_configs:
        for seed in seeds:
            param_key = (tuple(sorted(config.items())), seed)
            if param_key in completed_keys:
                print(f"Skipping (already completed): {config}, seed={seed}")
                continue

            result = run_single_experiment(
                params=config,
                seed=seed,
                gpu_id=gpu_id,
                output_base=search_dir,
                config_path="config.yaml",
                dry_run=dry_run,
            )
            group3_results.append(result)
            all_results.append(result)

            save_results(all_results, search_dir, "checkpoint.json")

    group3_agg = aggregate_results(group3_results)
    save_results(group3_agg, search_dir, "group3_aggregated.json")

    if group3_agg["results"]:
        best_final = group3_agg["results"][0]["params"]
        print(f"\nBest Final config: {best_final}")
        print(f"Balanced Acc: {group3_agg['results'][0]['stats'].get('balanced_accuracy_mean', 0):.2f}%")
    else:
        best_final = best_group2

    # =========================================================================
    # 最终汇总
    # =========================================================================
    final_summary = {
        "search_type": "grouped_grid",
        "timestamp": timestamp,
        "config": {
            "num_seeds": num_seeds,
            "gpu_id": gpu_id,
            "search_space": MOE_SEARCH_SPACE,
        },
        "group1_best": best_group1,
        "group2_best": best_group2,
        "group3_best": best_final,
        "final_best": best_final,
        "total_experiments": len(all_results),
        "successful_experiments": len([r for r in all_results if r["status"] == "success"]),
    }

    save_results(final_summary, search_dir, "final_summary.json")
    save_results(all_results, search_dir, "all_results.json")

    # 生成分析报告
    generate_analysis_report(search_dir, all_results, final_summary)

    print("\n" + "="*80)
    print("SEARCH COMPLETED!")
    print("="*80)
    print(f"Results directory: {search_dir}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {final_summary['successful_experiments']}")
    print(f"\nBest configuration:")
    for k, v in best_final.items():
        print(f"  {k}: {v}")

    return search_dir, best_final


def generate_analysis_report(search_dir: str, all_results: List[Dict], summary: Dict):
    """生成详细分析报告"""
    report_path = os.path.join(search_dir, "analysis_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# MoE Hyperparameter Search Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 概述
        f.write("## 1. Overview\n\n")
        f.write(f"- **Search Type**: {summary['search_type']}\n")
        f.write(f"- **Total Experiments**: {summary['total_experiments']}\n")
        f.write(f"- **Successful Experiments**: {summary['successful_experiments']}\n")
        f.write(f"- **Seeds per Config**: {summary['config']['num_seeds']}\n")
        f.write(f"- **GPU**: {summary['config']['gpu_id']}\n\n")

        # 搜索空间
        f.write("## 2. Search Space\n\n")
        f.write("| Parameter | Values |\n")
        f.write("|-----------|--------|\n")
        for param, values in summary['config']['search_space'].items():
            f.write(f"| {param} | {values} |\n")
        f.write("\n")

        # 最优配置
        f.write("## 3. Best Configuration\n\n")
        f.write("### Final Best Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for param, value in summary['final_best'].items():
            f.write(f"| {param} | {value} |\n")
        f.write("\n")

        # 各组最优
        f.write("### Group-wise Best\n\n")
        f.write("| Group | Best w_balance | Best gate_tau | Best scale | Best ldam_power | Best ldam_max_m | Best la_tau |\n")
        f.write("|-------|----------------|---------------|------------|-----------------|-----------------|-------------|\n")

        g1 = summary.get('group1_best', {})
        g2 = summary.get('group2_best', {})
        g3 = summary.get('group3_best', {})

        f.write(f"| Group 1 | {g1.get('w_balance', '-')} | {g1.get('gate_tau', '-')} | {g1.get('scale', '-')} | {g1.get('ldam_power', '-')} | {g1.get('ldam_max_m', '-')} | {g1.get('la_tau', '-')} |\n")
        f.write(f"| Group 2 | {g2.get('w_balance', '-')} | {g2.get('gate_tau', '-')} | {g2.get('scale', '-')} | {g2.get('ldam_power', '-')} | {g2.get('ldam_max_m', '-')} | {g2.get('la_tau', '-')} |\n")
        f.write(f"| Group 3 | {g3.get('w_balance', '-')} | {g3.get('gate_tau', '-')} | {g3.get('scale', '-')} | {g3.get('ldam_power', '-')} | {g3.get('ldam_max_m', '-')} | {g3.get('la_tau', '-')} |\n")
        f.write("\n")

        # 聚合结果分析
        f.write("## 4. Aggregated Results\n\n")

        agg = aggregate_results(all_results)

        if agg["results"]:
            f.write("### Top 10 Configurations by Balanced Accuracy\n\n")
            f.write("| Rank | w_balance | gate_tau | scale | ldam_power | ldam_max_m | la_tau | Bal.Acc (mean±std) | Minority Acc | HM |\n")
            f.write("|------|-----------|----------|-------|------------|------------|--------|-------------------|--------------|----|\n")

            for i, result in enumerate(agg["results"][:10], 1):
                p = result["params"]
                s = result["stats"]
                f.write(f"| {i} | {p['w_balance']} | {p['gate_tau']} | {p['scale']} | {p['ldam_power']} | {p['ldam_max_m']} | {p['la_tau']} | ")
                f.write(f"{s.get('balanced_accuracy_mean', 0):.2f}±{s.get('balanced_accuracy_std', 0):.2f} | ")
                f.write(f"{s.get('minority_acc_mean', 0):.2f} | ")
                f.write(f"{s.get('harmonic_mean_mean', 0):.2f} |\n")
            f.write("\n")

        # 参数敏感性分析
        f.write("## 5. Parameter Sensitivity Analysis\n\n")

        for param in MOE_SEARCH_SPACE.keys():
            f.write(f"### {param}\n\n")

            # 按参数值分组统计
            param_stats = {}
            for result in agg["results"]:
                val = result["params"][param]
                if val not in param_stats:
                    param_stats[val] = []
                param_stats[val].append(result["stats"].get("balanced_accuracy_mean", 0))

            f.write("| Value | Mean Balanced Acc | Std | Count |\n")
            f.write("|-------|-------------------|-----|-------|\n")
            for val in sorted(param_stats.keys()):
                accs = param_stats[val]
                f.write(f"| {val} | {np.mean(accs):.2f}% | {np.std(accs):.2f} | {len(accs)} |\n")
            f.write("\n")

        # 实验状态统计
        f.write("## 6. Experiment Status\n\n")
        status_counts = {}
        for r in all_results:
            status = r["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        f.write("| Status | Count |\n")
        f.write("|--------|-------|\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"| {status} | {count} |\n")
        f.write("\n")

        # 运行时间统计
        f.write("## 7. Runtime Statistics\n\n")
        durations = [r["duration"] for r in all_results if r["duration"]]
        if durations:
            f.write(f"- **Total Runtime**: {sum(durations)/3600:.2f} hours\n")
            f.write(f"- **Average per Experiment**: {np.mean(durations)/60:.2f} minutes\n")
            f.write(f"- **Min**: {np.min(durations)/60:.2f} minutes\n")
            f.write(f"- **Max**: {np.max(durations)/60:.2f} minutes\n")
        f.write("\n")

        f.write("---\n")
        f.write("*Report generated automatically by MoE hyperparameter search script.*\n")

    print(f"Analysis report saved to: {report_path}")
    return report_path


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MoE Hyperparameter Search")
    parser.add_argument("--gpu", type=str, default="3", help="GPU ID to use")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per config")
    parser.add_argument("--output", type=str, default="moe_search_results", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    parser.add_argument("--mode", type=str, default="grouped",
                       choices=["grouped", "full"],
                       help="Search mode: grouped (recommended) or full grid")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MoE Hyperparameter Search")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output}")
    print(f"Dry run: {args.dry_run}")
    print("="*80 + "\n")

    if args.mode == "grouped":
        run_grouped_search(
            gpu_id=args.gpu,
            num_seeds=args.seeds,
            output_dir=args.output,
            dry_run=args.dry_run,
            resume_from=args.resume,
        )
    else:
        # 完整网格搜索（警告：非常耗时）
        print("WARNING: Full grid search will take a very long time!")
        print(f"Total configs: {len(generate_all_configs())}")
        print(f"Total experiments: {len(generate_all_configs()) * args.seeds}")

        confirm = input("Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

        # 实现完整搜索...
        print("Full grid search not implemented. Use 'grouped' mode.")


if __name__ == "__main__":
    main()
