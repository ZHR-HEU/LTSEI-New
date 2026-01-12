# 长尾不平衡学习实验指南

本文档按照论文实验设计的逻辑组织，包含基准实验、对比实验、消融实验的完整命令行。

**MoE-LTSEI** 是本文提出的方法。

---

## 📋 实验设计概览

| 实验类型 | 目的 | 章节 |
|----------|------|------|
| **基准实验** | 建立 Baseline，验证长尾问题存在 | 1.1 |
| **单阶段对比实验** | 对比现有单阶段方法 | 2.1 - 2.3 |
| **两阶段对比实验** | 对比现有两阶段方法 | 3.1 - 3.3 |
| **消融实验** | 验证 MoE-LTSEI 各组件贡献 | 4 |
| **MoE-LTSEI (Ours)** | 本文提出的完整方法 | 5 |

---

## 1. 基准实验 (Baseline)

### 1.1 标准交叉熵 (无任何长尾处理)

作为所有对比实验的基准线。

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

---

## 2. 单阶段对比实验

### 2.1 重加权方法 (Re-weighting)

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

```bash
python main.py loss.name=LDAMLoss loss.ldam_max_margin=0.5 loss.ldam_scale=1.0 loss.ldam_drw_start=0 sampling.name=none stage2.enabled=false
```

#### LDAM + DRW (Deferred Re-Weighting)
> 同上，配合延迟重加权策略

```bash
python main.py loss.name=LDAMLoss loss.ldam_max_margin=0.5 loss.ldam_scale=1.0 loss.ldam_drw_start=160 loss.ldam_reweight_power=0.25 sampling.name=none stage2.enabled=false
```

---

### 2.2 先验调整方法 (Logit Adjustment)

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

```bash
python main.py loss.name=LogitAdjustmentLoss loss.logit_tau=1.0 sampling.name=none stage2.enabled=false
```

---

### 2.3 重采样方法 (Re-sampling)

#### 逆频率采样 (Inverse Frequency)
> 经典重采样方法，采样概率 $p_c \propto 1/n_c$

```bash
python main.py loss.name=CrossEntropy sampling.name=inv_freq stage2.enabled=false
```

#### 类别均匀采样 (Class-Uniform)
> 每个类别采样概率相等

```bash
python main.py loss.name=CrossEntropy sampling.name=class_uniform stage2.enabled=false
```

#### 平方根采样 (Square-Root)
> 采样概率 $p_c \propto 1/\sqrt{n_c}$，介于原始分布与均匀分布之间

```bash
python main.py loss.name=CrossEntropy sampling.name=sqrt stage2.enabled=false
```

---

## 3. 两阶段对比实验

### 3.1 Decoupling 方法

#### cRT (Classifier Re-Training)
> Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition", **ICLR 2020**
> 
> Stage-1: 标准训练学习表征；Stage-2: 冻结 backbone，平衡采样重训分类器

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.epochs=100 stage2.lr=0.1 stage2.optimizer=SGD stage2.loss=CrossEntropy stage2.sampler=class_uniform stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### τ-norm (Weight Normalization)
> 同上论文，对分类器权重做 τ 范数归一化

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=false stage3.mode=tau_norm stage3.tau_norm=1.0
```

#### LWS (Learnable Weight Scaling)
> 同上论文，使用可学习的类别权重缩放

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.epochs=100 stage2.lr=0.1 stage2.loss=CrossEntropy stage2.sampler=class_uniform stage2.freeze_bn=true
```

---

### 3.2 代价敏感两阶段方法

#### cRT + Cost-Sensitive CE
> Stage-2 使用代价敏感损失，代价权重 $w_c \propto 1/n_c$

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.epochs=100 stage2.lr=0.1 stage2.optimizer=SGD stage2.loss=CostSensitiveCE stage2.cost_strategy=auto stage2.sampler=progressive_power stage2.alpha_start=1.0 stage2.alpha_end=0.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### cRT + Cost-Sensitive CE (sqrt)
> 代价权重 $w_c \propto 1/\sqrt{n_c}$，更温和的重加权

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.epochs=100 stage2.lr=0.1 stage2.optimizer=SGD stage2.loss=CostSensitiveCE stage2.cost_strategy=sqrt stage2.sampler=progressive_power stage2.alpha_start=1.0 stage2.alpha_end=0.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

---

### 3.3 Label Smoothing 两阶段方法

#### cRT + LOS (Label Over-Smoothing)
> ICLR 2025，使用极大 Label Smoothing (ε≈0.98) 使目标分布接近均匀

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.epochs=100 stage2.lr=0.1 stage2.optimizer=SGD stage2.loss=LOS stage2.los_smoothing=0.98 stage2.sampler=class_uniform stage2.freeze_bn=true stage2.warmup_epochs=5
```

---

## 4. 消融实验 (Ablation Study)

验证 MoE-LTSEI 各组件的贡献。

### 4.1 Backbone 对比

#### 单专家: ConvNetADSB
```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### 单专家: ResNet1D
```bash
python main.py model.name=ResNet1D loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### 单专家: DilatedTCN
```bash
python main.py model.name=DilatedTCN loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### 单专家: FrequencyDomainExpert
```bash
python main.py model.name=FrequencyDomainExpert loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

#### MoE 结构 (无长尾处理)
```bash
python main.py model.name=MixtureOfExpertsConvNet loss.name=CrossEntropy sampling.name=none stage2.enabled=false
```

---

### 4.2 损失函数消融

#### MoE + 标准 CE (无边界)
```bash
python main.py model.name=MixtureOfExpertsConvNet loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.loss=CrossEntropy stage2.sampler=class_uniform stage2.freeze_bn=true stage2.epochs=100 stage2.lr=0.1
```

#### MoE + Class-Balanced Loss
```bash
python main.py model.name=MixtureOfExpertsConvNet loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.loss=ClassBalancedLoss stage2.sampler=class_uniform stage2.freeze_bn=true stage2.epochs=100 stage2.lr=0.1
```

#### MoE + Cost-Sensitive CE
```bash
python main.py model.name=MixtureOfExpertsConvNet loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=crt stage2.loss=CostSensitiveCE stage2.cost_strategy=auto stage2.sampler=progressive_power stage2.alpha_start=1.0 stage2.alpha_end=0.0 stage2.freeze_bn=true stage2.epochs=100 stage2.lr=0.1
```

---

### 4.3 MoE-LTSEI 组件消融

#### MoE-LTSEI w/o Gate Loss (λ_gate=0)
> 移除门控监督损失

```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe.gate_hidden=128 stage2.moe_loss.scale=30.0 stage2.moe_loss.beta=0.999 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=0.0 stage2.moe_loss.lambda_lb=0.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### MoE-LTSEI w/o Adaptive Margin (margin_m0=0)
> 移除自适应边界

```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe.gate_hidden=128 stage2.moe_loss.scale=30.0 stage2.moe_loss.beta=0.999 stage2.moe_loss.margin_m0=0.0 stage2.moe_loss.lambda_gate=1.0 stage2.moe_loss.lambda_lb=0.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### MoE-LTSEI w/o Difficulty Weighting (diff_gamma=0)
> 移除难度加权

```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe.gate_hidden=128 stage2.moe_loss.scale=30.0 stage2.moe_loss.beta=0.999 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.diff_gamma=0.0 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### MoE-LTSEI with Load Balance (λ_lb=0.01)
> 添加负载均衡正则化

```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe.gate_hidden=128 stage2.moe_loss.scale=30.0 stage2.moe_loss.beta=0.999 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.moe_loss.lambda_lb=0.01 stage2.freeze_bn=true stage2.warmup_epochs=5
```

---

### 4.4 专家数量消融

#### 2 Experts
```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=2 stage2.moe.gate_hidden=128 stage2.moe_loss.scale=30.0 stage2.moe_loss.beta=0.999 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### 3 Experts (默认)
```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe.gate_hidden=128 stage2.moe_loss.scale=30.0 stage2.moe_loss.beta=0.999 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

#### 4 Experts
```bash
python main.py model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=4 stage2.moe.gate_hidden=128 stage2.moe_loss.scale=30.0 stage2.moe_loss.beta=0.999 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true stage2.warmup_epochs=5
```

---

### 4.5 不平衡比消融

#### IR = 10
```bash
python main.py data.imbalance_ratio=10 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe_loss.scale=30.0 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true
```

#### IR = 50
```bash
python main.py data.imbalance_ratio=50 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe_loss.scale=30.0 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true
```

#### IR = 100 (默认)
```bash
python main.py data.imbalance_ratio=100 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe_loss.scale=30.0 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true
```

#### IR = 200
```bash
python main.py data.imbalance_ratio=200 model.name=ConvNetADSB loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe_loss.scale=30.0 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true
```

---

## 5. MoE-LTSEI (Ours) - 完整方法

### 5.1 完整配置

**Stage-1**: 标准 CE 训练，学习通用表征

**Stage-2**: MoE 分类器头 + 自适应边界损失 + 门控监督

```bash
python main.py \
  model.name=ConvNetADSB \
  model.dropout=0.1 \
  model.use_attention=true \
  loss.name=CrossEntropy \
  sampling.name=none \
  training.epochs=300 \
  training.lr=0.1 \
  training.optimizer=SGD \
  stage2.enabled=true \
  stage2.mode=moe_ltsei \
  stage2.epochs=100 \
  stage2.lr=0.1 \
  stage2.optimizer=SGD \
  stage2.loss=MoELTSEILoss \
  stage2.moe.num_experts=3 \
  stage2.moe.gate_hidden=128 \
  stage2.moe.gate_dropout=0.1 \
  stage2.moe.normalize_features=true \
  stage2.moe_loss.scale=30.0 \
  stage2.moe_loss.beta=0.999 \
  stage2.moe_loss.margin_m0=0.35 \
  stage2.moe_loss.margin_m1=0.2 \
  stage2.moe_loss.margin_gamma=0.5 \
  stage2.moe_loss.diff_gamma=2.0 \
  stage2.moe_loss.diff_alpha=1.0 \
  stage2.moe_loss.lambda_gate=1.0 \
  stage2.moe_loss.lambda_lb=0.0 \
  stage2.sampler=progressive_power \
  stage2.alpha_start=1.0 \
  stage2.alpha_end=0.0 \
  stage2.freeze_bn=true \
  stage2.warmup_epochs=5 \
  data.imbalance_ratio=100
```

### 5.2 简洁版本 (使用默认值)

```bash
python main.py loss.name=CrossEntropy sampling.name=none stage2.enabled=true stage2.mode=moe_ltsei stage2.epochs=100 stage2.lr=0.1 stage2.loss=MoELTSEILoss stage2.moe.num_experts=3 stage2.moe_loss.scale=30.0 stage2.moe_loss.margin_m0=0.35 stage2.moe_loss.lambda_gate=1.0 stage2.freeze_bn=true
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
| `stage2.mode` | 模式 | crt, finetune, moe_ltsei |
| `stage2.epochs` | Stage-2 训练轮次 | 50, 100 |
| `stage2.lr` | Stage-2 学习率 | 0.01, 0.1 |
| `stage2.freeze_bn` | 是否冻结 BN | true, false |

### D. MoE-LTSEI 专用参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `stage2.moe.num_experts` | 专家数量 | 3 |
| `stage2.moe.gate_hidden` | 门控隐藏层维度 | 128 |
| `stage2.moe_loss.scale` | Logit 缩放因子 | 30.0 |
| `stage2.moe_loss.beta` | 有效样本数 β | 0.999 |
| `stage2.moe_loss.margin_m0` | 基础边界 | 0.35 |
| `stage2.moe_loss.diff_gamma` | 难度调制 γ | 2.0 |
| `stage2.moe_loss.lambda_gate` | 门控损失权重 | 1.0 |
| `stage2.moe_loss.lambda_lb` | 负载均衡权重 | 0.0 |

---

## 📚 参考文献

1. **Focal Loss**: Lin et al., ICCV 2017
2. **Class-Balanced Loss**: Cui et al., CVPR 2019
3. **LDAM**: Cao et al., NeurIPS 2019
4. **Balanced Softmax**: Ren et al., NeurIPS 2020
5. **Logit Adjustment**: Menon et al., ICLR 2021
6. **Decoupling (cRT/τ-norm/LWS)**: Kang et al., ICLR 2020
7. **LOS**: ICLR 2025

---

*文档生成时间: 2026-01-12*
