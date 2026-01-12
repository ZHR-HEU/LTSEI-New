# -*- coding: utf-8 -*-
"""
Unified Training Module for Imbalanced Learning
================================================

本模块整合了所有训练相关的工具，包括：
- 随机种子与设备配置 (原 common.py)
- 学习率调度器 (原 training_utils.py)
- 早停与检查点 (原 training_utils.py)
- 优化器与损失函数构建 (原 optim_utils.py)
- 训练与评估循环 (原 train_eval.py)
- Stage-2 训练辅助 (原 stage2.py)
- 训练日志 (原 trainer_logging.py)

Author: ZHR
Date: 2025
"""

from __future__ import annotations
import os
import time
import random
from datetime import datetime
from typing import List, Optional, Any, Iterable, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from omegaconf import DictConfig

# =============================================================================
# Part 1: 随机种子 & 设备配置 (原 common.py)
# =============================================================================

def setup_seed(seed: int = 42):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def parse_gpu_ids(gpus_input) -> List[int]:
    """解析GPU ID配置"""
    if gpus_input is None or gpus_input == "null":
        return []
    if isinstance(gpus_input, int):
        return [gpus_input]
    if isinstance(gpus_input, str):
        if gpus_input.lower() in ["null", "none", ""]:
            return []
        parts = [p.strip() for p in gpus_input.split(',') if p.strip()]
        ids = []
        for p in parts:
            if not p.isdigit():
                raise ValueError(f"Invalid GPU id: {p}")
            ids.append(int(p))
        return ids
    if isinstance(gpus_input, (list, tuple)):
        return [int(x) for x in gpus_input]
    raise ValueError(f"Unsupported GPU input type: {type(gpus_input)}")


def setup_device(which: str = 'auto', gpu_ids: Optional[List[int]] = None) -> torch.device:
    """配置计算设备"""
    gpu_ids = gpu_ids or []
    if len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            print("CUDA not available, fallback to CPU despite --gpus provided.")
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cuda' if (which == 'auto' and torch.cuda.is_available()) else which)
    print(f"Using device: {device}")
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            if len(gpu_ids) > 0:
                for i in gpu_ids:
                    name = torch.cuda.get_device_name(i)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    print(f"  cuda:{i} -> {name} | VRAM: {total:.1f} GB")
            else:
                i = device.index or 0
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  cuda:{i} -> {name} | VRAM: {total:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        except Exception:
            pass
    return device


def convert_numpy_types(obj: Any) -> Any:
    """安全 JSON 序列化转换"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def logits_logit_adjustment(logits: torch.Tensor, class_counts: np.ndarray, tau: float = 1.0) -> torch.Tensor:
    """推理期Logit Adjustment后处理"""
    if tau <= 0:
        return logits
    prior = class_counts.astype(np.float64)
    prior = np.maximum(prior, 1.0)
    prior = prior / prior.sum()
    shift = torch.from_numpy(np.log(prior + 1e-12)).to(logits.device).float()
    return logits - tau * shift


def tau_norm_weights(weight: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """τ-norm权重归一化"""
    if tau == 0:
        return weight
    norm = weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    return weight / (norm ** tau)


# =============================================================================
# Part 2: 学习率调度器 (原 training_utils.py)
# =============================================================================

class GradualWarmupScheduler(LRScheduler):
    """
    渐进式学习率预热调度器
    
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1

        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                         for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class CosineAnnealingWarmupRestarts(LRScheduler):
    """带Warmup的余弦退火重启调度器"""

    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=1e-3, min_lr=1e-4,
                 warmup_steps=0, gamma=1.0):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            cos_steps = self.step_in_cycle - self.warmup_steps
            cos_total = self.cur_cycle_steps - self.warmup_steps
            return [base_lr + (self.max_lr - base_lr) *
                    (1 + np.cos(np.pi * cos_steps / cos_total)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.step_in_cycle == self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = 0
            self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
            self.max_lr *= self.gamma

        self.step_in_cycle += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# =============================================================================
# Part 3: 早停与检查点 (原 training_utils.py)
# =============================================================================

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',
                 trace_func=print, save_best_only=True, monitor='loss', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        if mode == 'min':
            self.val_best = np.inf
            self.is_better = lambda current, best: current < best - self.delta
        else:
            self.val_best = -np.inf
            self.is_better = lambda current, best: current > best + self.delta

    def __call__(self, val_metric, model, optimizer=None, epoch=None, **kwargs):
        current = val_metric

        if self.best_score is None:
            self.best_score = current
            self.save_checkpoint(current, model, optimizer, epoch, **kwargs)
        elif not self.is_better(current, self.best_score):
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.trace_func(f'Current {self.monitor}: {current:.6f}, Best {self.monitor}: {self.best_score:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current
            self.save_checkpoint(current, model, optimizer, epoch, **kwargs)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, optimizer=None, epoch=None, **kwargs):
        if self.verbose:
            direction = 'decreased' if self.mode == 'min' else 'increased'
            self.trace_func(
                f'Validation {self.monitor} {direction} ({self.val_best:.6f} --> {val_metric:.6f}). Saving model ...')

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_metric': val_metric,
            'epoch': epoch,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        checkpoint.update(kwargs)
        torch.save(checkpoint, self.path)
        self.val_best = val_metric


class ModelCheckpointer:
    """模型检查点管理器"""

    def __init__(self, save_dir='checkpoints', save_best=True, save_interval=0,
                 max_checkpoints=5, monitor='loss', mode='min'):
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.monitor = monitor
        self.mode = mode

        os.makedirs(save_dir, exist_ok=True)

        if mode == 'min':
            self.best_metric = np.inf
            self.is_better = lambda current, best: current < best
        else:
            self.best_metric = -np.inf
            self.is_better = lambda current, best: current > best

        self.saved_checkpoints = []

    def save(self, model, optimizer=None, epoch=None, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}

        save_as_best = False
        if self.save_best and self.monitor in metrics:
            current_metric = metrics[self.monitor]
            if self.is_better(current_metric, self.best_metric):
                self.best_metric = current_metric
                save_as_best = True

        save_interval = (self.save_interval > 0 and
                         epoch is not None and
                         epoch % self.save_interval == 0)

        if save_as_best or save_interval:
            timestamp = int(time.time())

            if save_as_best:
                filename = f'best_model_epoch_{epoch}_{timestamp}.pth'
            else:
                filename = f'checkpoint_epoch_{epoch}_{timestamp}.pth'

            filepath = os.path.join(self.save_dir, filename)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': timestamp,
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            checkpoint.update(kwargs)

            torch.save(checkpoint, filepath)
            self.saved_checkpoints.append(filepath)

            if len(self.saved_checkpoints) > self.max_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

            print(f"Checkpoint saved: {filepath}")


class MetricsTracker:
    """指标追踪器"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        self.current_metrics = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            self.current_metrics[key] = value

            if key not in self.best_metrics:
                self.best_metrics[key] = {'value': value, 'epoch': len(self.metrics[key]) - 1}
            else:
                if 'loss' in key.lower():
                    if value < self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': len(self.metrics[key]) - 1}
                else:
                    if value > self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': len(self.metrics[key]) - 1}

    def get_average(self, metric_name, last_n=None):
        if metric_name not in self.metrics:
            return None
        values = self.metrics[metric_name]
        if last_n is not None:
            values = values[-last_n:]
        return np.mean(values) if values else None

    def get_best(self, metric_name):
        return self.best_metrics.get(metric_name, None)

    def get_current(self, metric_name):
        return self.current_metrics.get(metric_name, None)

    def get_history(self, metric_name):
        return self.metrics.get(metric_name, [])

    def reset(self):
        self.metrics.clear()
        self.best_metrics.clear()
        self.current_metrics.clear()

    def summary(self):
        summary = {}
        for metric_name in self.metrics:
            summary[metric_name] = {
                'current': self.get_current(metric_name),
                'best': self.get_best(metric_name),
                'average_last_5': self.get_average(metric_name, 5),
                'total_epochs': len(self.metrics[metric_name])
            }
        return summary


class TrainingManager:
    """训练管理器：整合调度器、早停、检查点、指标追踪"""

    def __init__(self, model, optimizer, scheduler=None, early_stopping=None,
                 checkpointer=None, metrics_tracker=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpointer = checkpointer
        self.metrics_tracker = metrics_tracker or MetricsTracker()

        self.epoch = 0
        self.should_stop = False

    def step_epoch(self, metrics=None):
        if metrics is None:
            metrics = {}

        self.epoch += 1
        self.metrics_tracker.update(**metrics)

        if self.scheduler is not None:
            if isinstance(self.scheduler, GradualWarmupScheduler) and hasattr(self.scheduler,
                                                                              'after_scheduler') and isinstance(
                    self.scheduler.after_scheduler, ReduceLROnPlateau):
                monitor_metric = metrics.get('val_loss', metrics.get('loss', 0))
                self.scheduler.step_ReduceLROnPlateau(monitor_metric, self.epoch)
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                monitor_metric = metrics.get('val_loss', metrics.get('loss', 0))
                self.scheduler.step(monitor_metric)
            else:
                self.scheduler.step()

        if self.early_stopping is not None:
            monitor_metric = metrics.get(f'val_{self.early_stopping.monitor}',
                                         metrics.get(self.early_stopping.monitor, 0))
            self.early_stopping(monitor_metric, self.model, self.optimizer, self.epoch, **metrics)
            self.should_stop = self.early_stopping.early_stop

        if self.checkpointer is not None:
            self.checkpointer.save(self.model, self.optimizer, self.epoch, metrics)

    def get_current_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def should_stop_training(self):
        return self.should_stop

    def get_metrics_summary(self):
        return self.metrics_tracker.summary()


# =============================================================================
# Part 4: 工厂函数
# =============================================================================

def create_warmup_scheduler(optimizer, warmup_epochs, after_scheduler=None, multiplier=1.0):
    """创建Warmup调度器"""
    return GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=multiplier,
        total_epoch=warmup_epochs,
        after_scheduler=after_scheduler
    )


def create_early_stopping(patience=7, monitor='loss', mode='min', save_path='best_model.pth', verbose=True):
    """创建早停回调"""
    return EarlyStopping(
        patience=patience,
        monitor=monitor,
        mode=mode,
        path=save_path,
        verbose=verbose
    )


def create_training_manager(model, optimizer, warmup_epochs=5, patience=10,
                            save_dir='checkpoints', monitor='loss', after_scheduler=None):
    """创建完整的训练管理器"""
    scheduler = create_warmup_scheduler(optimizer, warmup_epochs, after_scheduler)
    early_stopping = create_early_stopping(patience=patience, monitor=monitor)
    checkpointer = ModelCheckpointer(save_dir=save_dir, monitor=monitor)
    metrics_tracker = MetricsTracker()

    return TrainingManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpointer=checkpointer,
        metrics_tracker=metrics_tracker
    )


# =============================================================================
# Part 5: 优化器与调度器构建 (原 optim_utils.py)
# =============================================================================

def build_scheduler_for_stage(optimizer, cfg: DictConfig, epochs_this_stage: int, stage: str = 'stage1'):
    """为指定训练阶段构建学习率调度器"""
    def pick(name: str):
        if stage == 'stage2' and hasattr(cfg.stage2, name) and getattr(cfg.stage2, name) is not None:
            return getattr(cfg.stage2, name)
        return getattr(cfg.scheduler, name)

    scheduler_name = pick('name') if stage == 'stage2' else cfg.scheduler.name
    after_scheduler = None
    scheduler = None

    if scheduler_name == 'cosine':
        after_scheduler = CosineAnnealingLR(optimizer, T_max=epochs_this_stage, eta_min=1e-6)
    elif scheduler_name == 'step':
        after_scheduler = StepLR(optimizer, step_size=pick('step_size'), gamma=pick('gamma'))
    elif scheduler_name == 'plateau':
        plateau_mode = 'min' if 'loss' in cfg.early_stopping.monitor else 'max'
        after_scheduler = ReduceLROnPlateau(
            optimizer, mode=plateau_mode,
            factor=pick('plateau_factor'), patience=pick('plateau_patience'),
            min_lr=pick('plateau_min_lr'))
    elif scheduler_name == 'cosine_warmup_restarts':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=pick('cos_first_cycle_steps'),
            cycle_mult=pick('cos_cycle_mult'),
            max_lr=pick('cos_max_lr') if pick('cos_max_lr') is not None else optimizer.param_groups[0]['lr'],
            min_lr=pick('cos_min_lr'),
            warmup_steps=pick('cos_warmup_steps'),
            gamma=pick('cos_gamma')
        )
    else:
        scheduler = None

    if scheduler_name != 'cosine_warmup_restarts':
        warmup_epochs = pick('warmup_epochs')
        warmup_mult = pick('warmup_multiplier')
        if after_scheduler is not None and warmup_epochs and warmup_epochs > 0:
            scheduler = create_warmup_scheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                after_scheduler=after_scheduler,
                multiplier=warmup_mult if warmup_mult is not None else 1.0
            )
        else:
            scheduler = after_scheduler
    return scheduler


def build_optimizer(optimizer_name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float):
    """构建优化器"""
    if optimizer_name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == 'SGD':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_optimizer_with_groups(optimizer_name: str, param_groups: List[dict], weight_decay: float):
    """构建带参数组的优化器"""
    if optimizer_name == 'SGD':
        return optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == 'Adam':
        return optim.Adam(param_groups, weight_decay=weight_decay)
    if optimizer_name == 'AdamW':
        return optim.AdamW(param_groups, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_criterion(loss_name: str, cfg: DictConfig, class_counts):
    """构建损失函数"""
    from losses import create_loss as get_loss_function
    
    loss_kwargs = {}

    # FocalLoss 参数
    if loss_name == 'FocalLoss':
        loss_kwargs['gamma'] = getattr(cfg.loss, 'focal_gamma', 2.0)
        if hasattr(cfg.loss, 'focal_alpha') and cfg.loss.focal_alpha is not None:
            loss_kwargs['alpha'] = cfg.loss.focal_alpha

    # CrossEntropy 参数
    if loss_name == 'CrossEntropy':
        if cfg.training.label_smoothing > 0:
            loss_kwargs['label_smoothing'] = cfg.training.label_smoothing
        if hasattr(cfg.loss, 'weight') and cfg.loss.weight is not None:
            loss_kwargs['weight'] = cfg.loss.weight

    # LOS (Label Over-Smoothing) parameters
    if loss_name == 'LOS':
        stage2_cfg = getattr(cfg, 'stage2', None)
        loss_cfg = getattr(cfg, 'loss', None)

        los_smoothing = None
        if stage2_cfg is not None:
            los_smoothing = getattr(stage2_cfg, 'los_smoothing', None)
        if los_smoothing is None and loss_cfg is not None:
            los_smoothing = getattr(loss_cfg, 'los_smoothing', None)
        loss_kwargs['smoothing'] = 0.98 if los_smoothing is None else los_smoothing

        los_true_prob = None
        if stage2_cfg is not None:
            los_true_prob = getattr(stage2_cfg, 'los_true_prob', None)
        if los_true_prob is None and loss_cfg is not None:
            los_true_prob = getattr(loss_cfg, 'los_true_prob', None)
        if los_true_prob is not None:
            loss_kwargs['true_prob'] = los_true_prob

        los_delta = None
        if stage2_cfg is not None:
            los_delta = getattr(stage2_cfg, 'los_delta', None)
        if los_delta is None and loss_cfg is not None:
            los_delta = getattr(loss_cfg, 'los_delta', None)
        if los_delta is not None:
            loss_kwargs['delta'] = los_delta

    # ClassBalancedLoss 参数
    if loss_name == 'ClassBalancedLoss':
        loss_kwargs['beta'] = getattr(cfg.loss, 'cb_beta', 0.9999)

    # LDAMLoss 参数
    # IMPORTANT: scale=1.0 for Linear classifier, scale=30.0 for Cosine classifier
    if loss_name == 'LDAMLoss':
        loss_kwargs['max_margin'] = getattr(cfg.loss, 'ldam_max_margin', 0.5)
        loss_kwargs['scale'] = getattr(cfg.loss, 'ldam_scale', 1.0)  # Default 1.0 for Linear
        loss_kwargs['drw_start_epoch'] = getattr(cfg.loss, 'ldam_drw_start', 0)
        loss_kwargs['reweight_power'] = getattr(cfg.loss, 'ldam_reweight_power', 0.25)
        loss_kwargs['use_normalized_margin'] = getattr(cfg.loss, 'ldam_use_normalized_margin', False)

    # LogitAdjustmentLoss 参数
    if loss_name == 'LogitAdjustmentLoss':
        loss_kwargs['tau'] = getattr(cfg.loss, 'logit_tau', 1.0)

    # ProgressiveLoss 参数
    if loss_name == 'ProgressiveLoss':
        loss_kwargs['total_epochs'] = cfg.training.epochs
        loss_kwargs['start_strategy'] = getattr(cfg.loss, 'prog_start_strategy', 'uniform')
        loss_kwargs['end_strategy'] = getattr(cfg.loss, 'prog_end_strategy', 'inverse')

    # Cost-Sensitive Losses 参数
    if loss_name in ['CostSensitiveCE', 'CostSensitiveExpected', 'CostSensitiveFocal']:
        stage2_cfg = getattr(cfg, 'stage2', None)
        loss_cfg = getattr(cfg, 'loss', None)

        cost_strategy = 'auto'
        if stage2_cfg and hasattr(stage2_cfg, 'cost_strategy'):
            cost_strategy = stage2_cfg.cost_strategy
        elif loss_cfg and hasattr(loss_cfg, 'cost_strategy'):
            cost_strategy = loss_cfg.cost_strategy

        loss_kwargs['cost_strategy'] = cost_strategy

        if cost_strategy == 'manual':
            if stage2_cfg and hasattr(stage2_cfg, 'cost_vector'):
                loss_kwargs['cost_vector'] = stage2_cfg.cost_vector
            elif stage2_cfg and hasattr(stage2_cfg, 'cost_matrix'):
                loss_kwargs['cost_matrix'] = stage2_cfg.cost_matrix
            elif loss_cfg and hasattr(loss_cfg, 'cost_vector'):
                loss_kwargs['cost_vector'] = loss_cfg.cost_vector
            elif loss_cfg and hasattr(loss_cfg, 'cost_matrix'):
                loss_kwargs['cost_matrix'] = loss_cfg.cost_matrix

        if cost_strategy in ['auto', 'sqrt', 'log']:
            loss_kwargs['class_counts'] = class_counts

        if loss_name == 'CostSensitiveFocal':
            gamma = 2.0
            if stage2_cfg and hasattr(stage2_cfg, 'focal_gamma'):
                gamma = stage2_cfg.focal_gamma
            elif loss_cfg and hasattr(loss_cfg, 'focal_gamma'):
                gamma = loss_cfg.focal_gamma
            loss_kwargs['gamma'] = gamma

    # 需要 class_counts 的损失函数
    LOSSES_NEED_COUNTS = {
        'ClassBalancedLoss',
        'LDAMLoss',
        'BalancedSoftmaxLoss',
        'LogitAdjustmentLoss',
        'ProgressiveLoss',
    }

    if loss_name in LOSSES_NEED_COUNTS:
        loss_kwargs['class_counts'] = class_counts

    # 创建并返回损失函数
    try:
        criterion = get_loss_function(loss_name, **loss_kwargs)
        if criterion is None:
            raise ValueError(f"get_loss_function returned None for {loss_name}")
        print(f"[Criterion] Successfully created {loss_name}")
        return criterion
    except Exception as e:
        print(f"[ERROR] Failed to create loss function '{loss_name}'")
        print(f"  Error: {e}")
        print(f"  loss_kwargs: {loss_kwargs}")
        raise


# =============================================================================
# Part 6: 训练与评估循环 (原 train_eval.py)
# =============================================================================

def _forward_model(model: nn.Module, x: torch.Tensor):
    """模型前向传播，统一处理输出格式"""
    out = model(x)
    if isinstance(out, tuple) and len(out) >= 1:
        logits = out[0]
        features = out[1] if len(out) > 1 else None
    else:
        logits, features = out, None
    return logits, features


def _compute_loss(criterion: nn.Module, logits: torch.Tensor, target: torch.Tensor, features: Optional[torch.Tensor] = None):
    """计算损失，自动处理不同损失函数的接口"""
    try:
        return criterion(logits, target, feature=features)
    except TypeError:
        return criterion(logits, target)


def _maybe_apply_eval_posthoc(logits: torch.Tensor, class_counts: np.ndarray, eval_mode: str, tau: float) -> torch.Tensor:
    """评估时可选的后处理"""
    return logits_logit_adjustment(logits, class_counts, tau=tau) if eval_mode == 'posthoc' else logits


def train_one_epoch(model, loader, criterion, optimizer, device, logger,
                    epoch: int, grad_clip: float = 0.0, use_amp: bool = False, 
                    scaler: Optional[torch.cuda.amp.GradScaler] = None):
    """单个epoch的训练"""
    model.train()
    if hasattr(criterion, "train"):
        criterion.train()
    if hasattr(criterion, "update_epoch"):
        criterion.update_epoch(epoch)
    run_loss, correct, total = 0.0, 0, 0
    scaler = scaler or torch.cuda.amp.GradScaler(enabled=False)
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            amp_enabled = bool(use_amp and device.type == "cuda")
            scaler = scaler or torch.cuda.amp.GradScaler(enabled=amp_enabled)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, features = _forward_model(model, x)
                loss = _compute_loss(criterion, logits, y, features)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, features = _forward_model(model, x)
            loss = _compute_loss(criterion, logits, y, features)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        run_loss += float(loss.item())
        _, pred = logits.max(1)
        total += y.size(0)
        correct += int(pred.eq(y).sum().item())
        
        if batch_idx % 50 == 0:
            logger.log_training_step(batch_idx, len(loader), loss.item(), 100.0 * correct / max(1, total),
                                     optimizer.param_groups[0]['lr'])
    
    avg_loss = run_loss / max(1, len(loader))
    avg_acc = 100.0 * correct / max(1, total)
    return {'loss': avg_loss, 'acc': avg_acc}


def evaluate_with_analysis(model, loader, criterion, device, analyzer,
                           class_counts: np.ndarray, eval_logit_adjust: str, eval_logit_tau: float):
    """带分析的评估"""
    model.eval()
    if hasattr(criterion, "eval"):
        criterion.eval()
    t0 = time.time()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []
    sm = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, features = _forward_model(model, x)
            logits_eval = _maybe_apply_eval_posthoc(logits, class_counts, eval_logit_adjust, eval_logit_tau)
            loss = _compute_loss(criterion, logits_eval, y, features)
            val_loss += float(loss.item())
            prob = sm(logits_eval).detach()
            _, pred = prob.max(1)
            total += y.size(0)
            correct += int(pred.eq(y).sum().item())
            all_probs.append(prob.cpu().numpy())
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
    
    avg_loss = val_loss / max(1, len(loader))
    avg_acc = 100.0 * correct / max(1, total)
    
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    probs = np.concatenate(all_probs, axis=0) if all_probs else None
    analysis = analyzer.analyze_predictions(np.array(all_targets), np.array(all_preds), prob=probs)
    elapsed = time.time() - t0
    timing = {
        'seconds': float(elapsed),
        'milliseconds': float(elapsed * 1000),
        'throughput_samples_per_sec': float(total / max(1e-9, elapsed))
    }
    return {'loss': avg_loss, 'acc': avg_acc, 'balanced_acc': balanced_acc}, analysis, all_preds, all_targets, timing


# =============================================================================
# Part 7: Stage-2 训练辅助 (原 stage2.py)
# =============================================================================

def get_base_model(model: nn.Module) -> nn.Module:
    """获取基础模型（处理DataParallel包装）"""
    return model.module if isinstance(model, nn.DataParallel) else model


def find_classifier_layers(model: nn.Module, num_classes: int) -> List[Tuple[str, nn.Module]]:
    """查找分类器层"""
    candidates: List[Tuple[str, nn.Module]] = []
    last_linear: Optional[Tuple[str, nn.Linear]] = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = (name, m)
            if m.out_features == num_classes:
                candidates.append((name, m))
    return candidates if candidates else ([last_linear] if last_linear is not None else [])


def _replace_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a submodule by its dotted name."""
    if not module_name:
        raise ValueError("module_name cannot be empty")
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def prepare_lws_stage2(model: nn.Module, num_classes: int, init_scale: float = 1.0):
    """
    Prepare LWS stage-2: replace classifier with LWS head and freeze all params
    except per-class scales.
    """
    from models import LearnableWeightScaling

    clf_pairs = [
        (name, layer)
        for name, layer in find_classifier_layers(model, num_classes)
        if isinstance(layer, nn.Linear) and layer.out_features == num_classes
    ]
    if not clf_pairs:
        raise ValueError("[Stage-2 LWS] Cannot find classifier layers to wrap")

    lws_modules = []
    for name, layer in clf_pairs:
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"[Stage-2 LWS] Layer '{name}' is not nn.Linear")
        lws = LearnableWeightScaling(layer, init_scale=init_scale)
        _replace_module_by_name(model, name, lws)
        lws_modules.append(lws)

    for p in model.parameters():
        p.requires_grad = False
    for lws in lws_modules:
        lws.log_scale.requires_grad = True

    return lws_modules


def freeze_backbone_params(model: nn.Module, classifier_names: Iterable[str]):
    """冻结backbone参数，只保留分类器可训练"""
    cls = list(classifier_names)
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(cn) for cn in cls)


def unfreeze_all_params(model: nn.Module):
    """解冻所有参数"""
    for p in model.parameters():
        p.requires_grad = True


def reinit_classifier_layers(layers: List[nn.Module]):
    """重新初始化分类器层"""
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def apply_tau_norm_to_classifier(layers: List[nn.Module], tau: float = 1.0):
    """对分类器层应用τ-norm"""
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(tau_norm_weights(layer.weight, tau=tau))


def set_batchnorm_eval(module: nn.Module):
    """设置BatchNorm为评估模式"""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        module.eval()


def build_stage2_loader(dataset, batch_size: int, num_workers: int, sampler_config: Dict,
                        total_epochs: int = 100, seed: int = 0):
    """构建Stage-2数据加载器"""
    from data import make_sampler

    if not hasattr(dataset, 'labels'):
        raise AttributeError("Stage-2 sampler requires dataset.labels")

    labels = np.asarray(dataset.labels).astype(int)
    sampler_name = sampler_config.get('name', 'none')

    if sampler_name == 'same':
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    sampler = make_sampler(
        labels=labels,
        method=sampler_name,
        seed=seed,
        alpha=sampler_config.get('alpha', 0.5),
        alpha_start=sampler_config.get('alpha_start', 0.5),
        alpha_end=sampler_config.get('alpha_end', 0.0),
        total_epochs=total_epochs
    )

    if sampler is None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )


# =============================================================================
# Part 8: 训练日志 (原 trainer_logging.py)
# =============================================================================

def _format_ms(seconds):
    return f"{seconds * 1000:.2f} ms"


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_file: str, console_interval: int = 1):
        self.log_file = log_file
        self.console_interval = console_interval
        self.epoch_start_time = None
        self.per_epoch_times = []
        with open(log_file, 'w') as f:
            f.write("epoch,timestamp,train_loss,train_acc,val_loss,val_acc,val_balanced_acc,"
                    "majority_acc,medium_acc,minority_acc,lr,epoch_time\n")

    def start_epoch(self, epoch: int, lr: float = None):
        self.epoch_start_time = time.time()
        if epoch % self.console_interval == 0:
            lr_info = f" | LR: {lr:.2e}" if lr is not None else ""
            print(f"\n{'=' * 80}\nEpoch {epoch}{lr_info}\n{'=' * 80}")

    def log_training_step(self, batch_idx: int, total_batches: int, loss: float, acc: float, lr: float):
        if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  [Train] Batch {batch_idx + 1:4d}/{total_batches} ({progress:5.1f}%) | "
                  f"Loss: {loss:.6f} | Acc: {acc:5.2f}% | LR: {lr:.2e}")

    def log_epoch_end(self, epoch: int, train_metrics, val_metrics, group_metrics, lr: float):
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if epoch % self.console_interval == 0:
            print(f"\n{'─' * 80}")
            print(f"Epoch {epoch} Summary | Learning Rate: {lr:.2e}")
            print(f"  Time: {_format_ms(epoch_time)}")
            print(f"  Train - Loss: {train_metrics['loss']:.6f} | Acc: {train_metrics['acc']:5.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f} | Acc: {val_metrics['acc']:5.2f}% | "
                  f"Bal Acc: {val_metrics['balanced_acc']:5.2f}%")
            if group_metrics:
                print("  Class Groups:")
                for g, m in group_metrics.items():
                    if m:
                        print(f"    {g.capitalize():8s}: Acc={m['accuracy']:5.2f}% | F1={m['f1']:5.2f}% | Support={m['support']:4d}")

        with open(self.log_file, 'a') as f:
            maj = group_metrics.get('majority', {}).get('accuracy', 0)
            med = group_metrics.get('medium', {}).get('accuracy', 0)
            mino = group_metrics.get('minority', {}).get('accuracy', 0)
            f.write(f"{epoch},{ts},{train_metrics['loss']:.6f},{train_metrics['acc']:.4f},"
                    f"{val_metrics['loss']:.6f},{val_metrics['acc']:.4f},{val_metrics['balanced_acc']:.4f},"
                    f"{maj:.4f},{med:.4f},{mino:.4f},{lr:.8f},{epoch_time:.2f}\n")
        self.per_epoch_times.append(epoch_time)
