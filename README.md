

本文档以"实验目标 + 干预位置"为主线组织实验：先建立基准，再按"损失/先验/采样/分类器再平衡/后处理"进行对比。

---

## 实验设计概览

| 实验块 | 目的 | 章节 |
|----------|------|------|
| **基准与问题确认** | 建立 Baseline，确认长尾影响 | 2 |
| **方法对比（按干预位置）** | 对比主流长尾策略 | 3 |
| **消融与鲁棒性** | 验证组件贡献与稳健性 | 4 |

---

## 1. 实验设计与评估

### 1.1 统一设置
- 主干模型默认 `ConvNetADSB`，训练超参默认沿用 `config.yaml`。
- 仅在方法需要时启用 Stage-2（分类器再平衡）或 Stage-3（后处理）。
- 长尾由 `create_imbalance=true` 与 `data.imbalance_ratio` 控制。
- 建议固定随机种子并多次复现实验（如 3-5 个 seed）。

### 1.2 评估指标
- Overall Acc、Balanced Acc、Macro-F1、mAcc (Mean Per-Class Acc)。
- 头/中/尾分组性能（Many/Medium/Few）。

---

## 2. 基准与问题确认

### 2.1 标准交叉熵 (Baseline)
作为所有对比实验的基准线。

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

---

## 3. 方法对比（按干预位置）

### 3.1 损失函数级重加权 (Re-weighting)

#### Focal Loss
> Lin et al., "Focal Loss for Dense Object Detection", **ICCV 2017**
>
> 通过 $(1-p_t)^\gamma$ 调制因子聚焦难分类样本

```bash
python main.py loss.name=FocalLoss loss.focal_gamma=2.0 sampling.name=none stage2.enabled=false
```

#### Class-Balanced Loss (CB)
> Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", **CVPR 2019**
>
> 基于有效样本数 $E_n = (1-\beta^n)/(1-\beta)$ 计算类别权重

```bash
python main.py loss.name=ClassBalancedLoss loss.cb_beta=0.9999 sampling.name=none stage2.enabled=false
```

#### LDAM (Label-Distribution-Aware Margin)
> Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", **NeurIPS 2019**
>
> 为少数类设置更大边界 $\Delta_j = C / n_j^{1/4}$
> 说明：`ldam_reweight_power=0` 等效关闭 DRW

```bash
python main.py loss.name=LDAMLoss +loss.ldam_max_margin=0.5 +loss.ldam_scale=1.0 +loss.ldam_drw_start=0 +loss.ldam_reweight_power=0.0 sampling.name=none stage2.enabled=false
```

#### LDAM + DRW (Deferred Re-Weighting)
> 同上，配合延迟重加权策略

```bash
python main.py loss.name=LDAMLoss +loss.ldam_max_margin=0.5 +loss.ldam_scale=1.0 +loss.ldam_drw_start=120 +loss.ldam_reweight_power=0.25 sampling.name=none stage2.enabled=false
```

---

### 3.2 先验/Logit 调整 (Prior Adjustment)

#### Balanced Softmax
> Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", **NeurIPS 2020**
>
> Logits 加入类别先验 $\tilde{z}_c = z_c + \log \pi_c$

```bash
python main.py loss.name=BalancedSoftmaxLoss sampling.name=none stage2.enabled=false
```

#### Logit Adjustment (LA)
> Menon et al., "Long-tail Learning via Logit Adjustment", **ICLR 2021**
>
> 可调强度的先验调整 $\tilde{z}_c = z_c + \tau \log \pi_c$
> 注意：避免与模型头中的 Logit-Adjustment 同时启用

```bash
python main.py loss.name=LogitAdjustmentLoss +loss.logit_tau=1.0 sampling.name=none stage2.enabled=false
```

---

### 3.3 数据采样级 (Re-sampling)

#### 逆频率采样 (Inverse Frequency)
> 经典重采样方法，采样概率 $p_c \propto 1/n_c$
> 参考：Cui et al., **CVPR 2019**（Class-Balanced Loss 作为重采样/重加权基线）

```bash
python main.py loss.name=CrossEntropy sampling.name=inv_freq stage2.enabled=false
```

#### 类别均匀采样 (Class-Uniform)
> 每个类别采样概率相等
> 参考：Kang et al., **ICLR 2020**（cRT 的 class-balanced sampling）；
> Yang et al., **NeurIPS 2023**（How Re-sampling Helps for Long-Tail Learning?）

```bash
python main.py loss.name=CrossEntropy sampling.name=class_uniform stage2.enabled=false
```

#### 平方根采样 (Square-Root)
> 采样概率 $p_c \propto 1/\sqrt{n_c}$，介于原始分布与均匀分布之间
> 参考：Gupta et al., **CVPR 2019**（LVIS Repeat-Factor Sampling 思路）

```bash
python main.py loss.name=CrossEntropy sampling.name=sqrt stage2.enabled=false
```

---

### 3.4 分类器再平衡/解耦 (Classifier Rebalancing)

#### cRT (Classifier Re-Training)
> Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition", **ICLR 2020**
>
> Stage-1: 标准训练学习表征；Stage-2: 冻结 backbone，平衡采样重训分类器

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.epochs=100 stage2.lr=0.1 stage2.optimizer=SGD stage2.loss=CrossEntropy stage2.sampler=class_uniform stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### LWS (Learnable Weight Scaling)
> 同上论文，冻结分类器权重，仅学习每类缩放参数

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=lws stage2.epochs=100 stage2.lr=0.1 stage2.optimizer=SGD stage2.loss=CrossEntropy stage2.sampler=class_uniform stage2.lws_init_scale=1.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### LOS (Label Over-Smoothing)
> ICLR 2025，使用极大 Label Smoothing (ε≈0.98) 使目标分布接近均匀

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.epochs=100 stage2.lr=0.1 stage2.optimizer=SGD stage2.loss=LOS +stage2.los_smoothing=0.98 stage2.sampler=class_uniform stage2.freeze_bn=true stage2.warmup_epochs=5
```

---

### 3.5 后处理/校准 (Post-hoc)

#### τ-norm (Weight Normalization)
> 对分类器权重做 τ 范数归一化

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=false stage3.mode=tau_norm stage3.tau_norm=1.0
```

---

## 4. 消融与鲁棒性 (Ablation & Robustness)

### 4.1 Backbone 对比

#### ConvNetADSB
```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### ResNet1D
```bash
python main.py model.name=ResNet1D loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### DilatedTCN
```bash
python main.py model.name=DilatedTCN loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

---

### 4.2 不平衡比消融

#### IR = 10
```bash
python main.py data.imbalance_ratio=10 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### IR = 50
```bash
python main.py data.imbalance_ratio=50 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### IR = 100 (默认)
```bash
python main.py data.imbalance_ratio=100 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### IR = 200
```bash
python main.py data.imbalance_ratio=200 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

---

## 附录: 通用参数说明

### A. 数据相关
| 参数 | 说明 | 示例 |
|------|------|------|
| `data.imbalance_ratio` | 不平衡比 (头类/尾类) | 10, 50, 100, 200 |
| `data.batch_size` | 批量大小 | 128, 256 |

### B. 训练相关
| 参数 | 说明 | 示例 |
|------|------|------|
| `training.epochs` | Stage-1 训练轮次 | 200, 300 |
| `training.lr` | Stage-1 学习率 | 0.01, 0.1 |
| `training.optimizer` | 优化器 | SGD, Adam, AdamW |
| `gpus` | GPU ID | 0, "0,1" |

### C. Stage-2 相关
| 参数 | 说明 | 示例 |
|------|------|------|
| `stage2.enabled` | 是否启用 Stage-2 | true, false |
| `stage2.mode` | 模式 | crt, lws, finetune |
| `stage2.epochs` | Stage-2 训练轮次 | 50, 100 |
| `stage2.lr` | Stage-2 学习率 | 0.01, 0.1 |
| `stage2.lws_init_scale` | LWS 缩放初值 | 1.0 |
| `stage2.freeze_bn` | 是否冻结 BN | true, false |

---

## 参考文献

1. **Focal Loss**: Lin et al., ICCV 2017
2. **Class-Balanced Loss**: Cui et al., CVPR 2019
3. **LDAM**: Cao et al., NeurIPS 2019
4. **Balanced Softmax**: Ren et al., NeurIPS 2020
5. **Logit Adjustment**: Menon et al., ICLR 2021
6. **Decoupling (cRT/τ-norm/LWS)**: Kang et al., ICLR 2020
7. **LOS**: ICLR 2025
8. **LVIS Repeat-Factor Sampling**: Gupta et al., CVPR 2019
9. **Re-sampling Analysis**: Yang et al., NeurIPS 2023

---

*文档生成时间: 2026-01-12*
