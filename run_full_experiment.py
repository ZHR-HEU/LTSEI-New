#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行完整实验流程

该脚本将自动：
1. 运行所有方法（CE, LDAM, FocalLoss, CB, CRT, LOS, LTSEI）
2. 每种方法运行10个随机种子
3. 收集所有结果
4. 计算均值和标准差
5. 生成带误差棒的对比图
6. 生成LaTeX和Markdown表格

Usage:
    python run_full_experiment.py                    # 完整运行（所有方法，10个种子）
    python run_full_experiment.py --seeds 3          # 快速测试（3个种子）
    python run_full_experiment.py --methods CE LDAM  # 只运行指定方法
    python run_full_experiment.py --skip_training    # 跳过训练，只分析
    python run_full_experiment.py --gpu 0            # 指定GPU
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def run_command(cmd: list, description: str = ""):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    if description:
        print(f"[执行] {description}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="一键运行完整实验流程")
    parser.add_argument("--methods", nargs="+", default=None,
                       help="要运行的方法列表（默认全部）")
    parser.add_argument("--seeds", type=int, default=10,
                       help="每种方法的种子数量（默认10）")
    parser.add_argument("--seed_start", type=int, default=42,
                       help="起始种子（默认42）")
    parser.add_argument("--gpu", type=str, default="3",
                       help="GPU ID（默认3）")
    parser.add_argument("--skip_training", action="store_true",
                       help="跳过训练，只运行分析")
    parser.add_argument("--dry_run", action="store_true",
                       help="只显示命令，不实际执行")
    parser.add_argument("--output_dir", type=str, default="results_analysis",
                       help="分析结果输出目录")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "#"*80)
    print("# 完整实验流程")
    print("#"*80)
    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# GPU: {args.gpu}")
    print(f"# 种子数量: {args.seeds}")
    print(f"# 方法: {args.methods or '全部'}")
    print("#"*80 + "\n")
    
    # Step 1: 运行实验
    if not args.skip_training:
        print("\n" + "="*80)
        print("Step 1: 运行批量实验")
        print("="*80)
        
        train_cmd = [
            sys.executable, "run_all_experiments.py",
            "--seeds", str(args.seeds),
            "--seed_start", str(args.seed_start),
            "--gpu", args.gpu,
        ]
        
        if args.methods:
            train_cmd.extend(["--methods"] + args.methods)
        
        if args.dry_run:
            train_cmd.append("--dry_run")
        
        success = run_command(train_cmd, "批量训练所有方法")
        
        if not success and not args.dry_run:
            print("\n警告: 部分实验可能失败，继续进行分析...")
    else:
        print("\n跳过训练步骤（--skip_training）")
    
    # Step 2: 分析结果
    print("\n" + "="*80)
    print("Step 2: 分析实验结果")
    print("="*80)
    
    analyze_cmd = [
        sys.executable, "analyze_results.py",
        "--output_dir", f"{args.output_dir}_{timestamp}",
    ]
    
    success = run_command(analyze_cmd, "生成分析报告和图表")
    
    # 完成
    print("\n" + "#"*80)
    print("# 实验流程完成！")
    print("#"*80)
    print(f"# 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 分析结果保存在: {args.output_dir}_{timestamp}/")
    print("#"*80 + "\n")
    
    # 显示生成的文件列表
    output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / f"{args.output_dir}_{timestamp}"
    if output_dir.exists():
        print("生成的文件：")
        for f in sorted(output_dir.iterdir()):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
