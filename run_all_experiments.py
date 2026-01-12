#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量实验运行脚本 - 运行所有不平衡学习方法

支持的方法：
- CE: 标准交叉熵损失（baseline）
- LDAM: Label-Distribution-Aware Margin Loss
- FocalLoss: Focal Loss
- CB: Class-Balanced Loss
- CRT: Classifier Re-Training
- LOS: Label Over-Smoothing
- LTSEI: 我们的方法 (MoE-LTSEI)

每种方法运行多个种子，自动收集结果，生成图表和表格。

Usage:
    python run_all_experiments.py                    # 运行所有实验
    python run_all_experiments.py --methods CE LDAM  # 只运行指定方法
    python run_all_experiments.py --seeds 3          # 只运行3个种子
    python run_all_experiments.py --analyze_only     # 只分析已有结果
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time


# =============================================================================
# 实验方法定义
# =============================================================================

# 每种方法的Hydra配置覆盖参数
METHOD_CONFIGS = {
    # 1. CE - 标准交叉熵（Baseline）
    "CE": {
        "description": "Cross Entropy (Baseline)",
        "overrides": [
            "loss.name=CrossEntropy",
            "sampling.name=none",
            "stage2.enabled=false",
        ],
    },
    
    # 2. LDAM - Label-Distribution-Aware Margin Loss
    "LDAM": {
        "description": "LDAM with DRW",
        "overrides": [
            "loss.name=LDAMLoss",
            "+loss.ldam_max_margin=0.5",
            "+loss.ldam_scale=1.0",
            "+loss.ldam_drw_start=120",
            "+loss.ldam_reweight_power=0.25",
            "stage2.enabled=false",
        ],
    },
    
    # 3. FocalLoss - Focal Loss
    "FocalLoss": {
        "description": "Focal Loss",
        "overrides": [
            "loss.name=FocalLoss",
            "+loss.focal_gamma=2.0",
            "+loss.focal_alpha=0",
            "stage2.enabled=false",
        ],
    },
    
    # 4. CB - Class-Balanced Loss
    "CB": {
        "description": "Class-Balanced Loss",
        "overrides": [
            "loss.name=ClassBalancedLoss",
            "+loss.cb_beta=0.9999",
            "stage2.enabled=false",
            "sampling.name=none",
        ],
    },
    
    # 5. CRT - Classifier Re-Training
    "CRT": {
        "description": "Classifier Re-Training",
        "overrides": [
            "loss.name=CrossEntropy",
            "stage2.enabled=true",
            "stage2.mode=crt",
            "stage2.loss=CrossEntropy",
            "stage2.sampler=class_uniform",
            "sampling.name=none",
        ],
    },
    
    # 6. LOS - Label Over-Smoothing
    "LOS": {
        "description": "Label Over-Smoothing (Stage-2)",
        "overrides": [
            "loss.name=CrossEntropy",
            "stage2.enabled=true",
            "stage2.mode=crt",
            "stage2.loss=LOS",
            "+stage2.los_smoothing=0.99",
            "stage2.sampler=class_uniform",
            "sampling.name=none",
        ],
    },
    
    # 7. LTSEI - 我们的方法（MoE-LTSEI）
    "LTSEI": {
        "description": "MoE-LTSEI (Ours)",
        "overrides": [
            # 使用默认配置（config.yaml中已配置）
            # 这里可以添加额外的覆盖参数
        ],
    },
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_experiment(
    method_name: str,
    seed: int,
    gpu: str = "0",
    extra_overrides: Optional[List[str]] = None,
    dry_run: bool = False,
    workspace: str = "."
) -> Dict[str, Any]:
    """
    运行单个实验
    
    Args:
        method_name: 方法名称
        seed: 随机种子
        gpu: GPU ID
        extra_overrides: 额外的Hydra覆盖参数
        dry_run: 是否只打印命令而不执行
        workspace: 工作目录
        
    Returns:
        实验结果字典
    """
    if method_name not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_CONFIGS.keys())}")
    
    config = METHOD_CONFIGS[method_name]
    
    # 构建实验名称
    exp_name = f"{method_name}_seed{seed}"
    
    # 构建命令
    cmd = ["python", "main.py"]
    cmd.extend([f"exp_name={exp_name}"])
    cmd.extend([f"seed={seed}"])
    cmd.extend([f"gpus={gpu}"])
    
    # 添加方法特定的覆盖参数
    cmd.extend(config["overrides"])
    
    # 添加额外的覆盖参数
    if extra_overrides:
        cmd.extend(extra_overrides)
    
    print(f"\n{'='*80}")
    print(f"[{method_name}] Seed={seed} | {config['description']}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return {
            "method": method_name,
            "seed": seed,
            "status": "dry_run",
            "exp_name": exp_name,
        }
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行实验
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[SUCCESS] Completed in {elapsed/60:.1f} minutes")
            
            # 尝试查找结果文件
            exp_dir = find_experiment_dir(exp_name, workspace)
            results = load_experiment_results(exp_dir) if exp_dir else None
            
            return {
                "method": method_name,
                "seed": seed,
                "status": "success",
                "exp_name": exp_name,
                "exp_dir": str(exp_dir) if exp_dir else None,
                "elapsed_seconds": elapsed,
                "results": results,
            }
        else:
            print(f"[FAILED] Return code: {result.returncode}")
            print(f"stderr: {result.stderr[:500]}...")
            return {
                "method": method_name,
                "seed": seed,
                "status": "failed",
                "exp_name": exp_name,
                "error": result.stderr[:1000],
            }
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Experiment exceeded time limit")
        return {
            "method": method_name,
            "seed": seed,
            "status": "timeout",
            "exp_name": exp_name,
        }
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            "method": method_name,
            "seed": seed,
            "status": "error",
            "exp_name": exp_name,
            "error": str(e),
        }


def find_experiment_dir(exp_name: str, workspace: str = ".") -> Optional[Path]:
    """查找实验目录（找最新的匹配目录）"""
    experiments_dir = Path(workspace) / "experiments"
    if not experiments_dir.exists():
        return None
    
    # 查找以exp_name开头的目录
    matching_dirs = list(experiments_dir.glob(f"{exp_name}_*"))
    if not matching_dirs:
        return None
    
    # 返回最新的目录
    return max(matching_dirs, key=lambda p: p.stat().st_mtime)


def load_experiment_results(exp_dir: Path) -> Optional[Dict]:
    """加载实验结果"""
    results_file = exp_dir / "results" / "results.json"
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load results from {results_file}: {e}")
        return None


def run_all_experiments(
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    gpu: str = "0",
    dry_run: bool = False,
    workspace: str = ".",
    extra_overrides: Optional[List[str]] = None,
) -> List[Dict]:
    """
    运行所有实验
    
    Args:
        methods: 要运行的方法列表（None表示全部）
        seeds: 种子列表（None表示默认10个种子）
        gpu: GPU ID
        dry_run: 是否只打印命令
        workspace: 工作目录
        extra_overrides: 额外的覆盖参数
        
    Returns:
        所有实验结果列表
    """
    if methods is None:
        methods = list(METHOD_CONFIGS.keys())
    
    if seeds is None:
        seeds = list(range(42, 52))  # 默认10个种子: 42-51
    
    all_results = []
    total_experiments = len(methods) * len(seeds)
    current = 0
    
    print(f"\n{'#'*80}")
    print(f"# 批量实验开始")
    print(f"# 方法: {methods}")
    print(f"# 种子: {seeds}")
    print(f"# 总计: {total_experiments} 个实验")
    print(f"{'#'*80}")
    
    for method in methods:
        for seed in seeds:
            current += 1
            print(f"\n[{current}/{total_experiments}]", end=" ")
            
            result = run_experiment(
                method_name=method,
                seed=seed,
                gpu=gpu,
                extra_overrides=extra_overrides,
                dry_run=dry_run,
                workspace=workspace,
            )
            all_results.append(result)
    
    # 保存所有结果
    save_batch_results(all_results, workspace)
    
    return all_results


def save_batch_results(results: List[Dict], workspace: str = "."):
    """保存批量实验结果"""
    output_dir = Path(workspace) / "batch_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = get_timestamp()
    output_file = output_dir / f"batch_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n批量结果已保存: {output_file}")


def collect_existing_results(workspace: str = ".") -> List[Dict]:
    """收集已有的实验结果"""
    experiments_dir = Path(workspace) / "experiments"
    if not experiments_dir.exists():
        print("No experiments directory found")
        return []
    
    all_results = []
    
    for method_name in METHOD_CONFIGS.keys():
        for exp_dir in experiments_dir.glob(f"{method_name}_seed*_*"):
            # 解析种子
            dir_name = exp_dir.name
            try:
                seed_part = dir_name.split("_seed")[1].split("_")[0]
                seed = int(seed_part)
            except:
                continue
            
            results = load_experiment_results(exp_dir)
            if results:
                all_results.append({
                    "method": method_name,
                    "seed": seed,
                    "status": "success",
                    "exp_name": dir_name.rsplit("_", 1)[0],  # 去掉时间戳
                    "exp_dir": str(exp_dir),
                    "results": results,
                })
    
    return all_results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="批量运行不平衡学习实验")
    parser.add_argument("--methods", nargs="+", default=None,
                       help=f"要运行的方法列表。可选: {list(METHOD_CONFIGS.keys())}")
    parser.add_argument("--seeds", type=int, default=10,
                       help="种子数量（默认10，使用种子42-51）")
    parser.add_argument("--seed_start", type=int, default=42,
                       help="起始种子（默认42）")
    parser.add_argument("--gpu", type=str, default="3",
                       help="GPU ID（默认3）")
    parser.add_argument("--dry_run", action="store_true",
                       help="只打印命令，不实际执行")
    parser.add_argument("--analyze_only", action="store_true",
                       help="只分析已有结果，不运行实验")
    parser.add_argument("--extra", nargs="*", default=None,
                       help="额外的Hydra覆盖参数")
    
    args = parser.parse_args()
    
    workspace = os.path.dirname(os.path.abspath(__file__))
    
    # 生成种子列表
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    
    if args.analyze_only:
        print("收集已有实验结果...")
        results = collect_existing_results(workspace)
        if results:
            save_batch_results(results, workspace)
            print(f"找到 {len(results)} 个实验结果")
        else:
            print("未找到实验结果")
    else:
        # 运行实验
        results = run_all_experiments(
            methods=args.methods,
            seeds=seeds,
            gpu=args.gpu,
            dry_run=args.dry_run,
            workspace=workspace,
            extra_overrides=args.extra,
        )
        
        # 打印摘要
        print("\n" + "="*80)
        print("实验摘要")
        print("="*80)
        
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        
        print(f"成功: {success_count}/{len(results)}")
        print(f"失败: {failed_count}/{len(results)}")
        
        if failed_count > 0:
            print("\n失败的实验:")
            for r in results:
                if r["status"] == "failed":
                    print(f"  - {r['method']} seed={r['seed']}")


if __name__ == "__main__":
    main()
