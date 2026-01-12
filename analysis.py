# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# ============================================================================
# 从 sklearn.metrics 导入各种评估指标函数
# ============================================================================
from sklearn.metrics import (
    confusion_matrix,              # 混淆矩阵：展示预测结果与真实标签的对应关系
    balanced_accuracy_score,       # 平衡准确率：各类别召回率的算术平均，对不平衡数据更公平
    precision_recall_fscore_support,  # 计算精确率、召回率、F1分数和支持度
    top_k_accuracy_score,          # Top-K准确率：真实类别在预测概率前K个中的比例
    roc_auc_score,                 # ROC曲线下面积：衡量分类器的排序能力
    average_precision_score        # 平均精确率：PR曲线下面积的近似
)

# ============================================================================
# 可选的绘图库导入
# 如果 matplotlib 或 seaborn 不可用，则禁用绘图功能
# ============================================================================
try:
    import matplotlib.pyplot as plt  # 基础绘图库
    import seaborn as sns            # 高级统计绘图库，用于绘制热力图
    HAS_PLOTTING = True              # 绘图功能可用标志
except Exception:
    HAS_PLOTTING = False             # 绘图功能不可用


class ClassificationAnalyzer:
   
    
    def __init__(
        self,
        class_counts: np.ndarray,
        class_names: Optional[List[str]] = None,
        grouping: str = "auto",
        many_thresh: int = 100,
        few_thresh: int = 20,
        q_low: float = 1 / 3,
        q_high: float = 2 / 3
    ):
       
        # 将输入转换为整数类型的 numpy 数组
        self.class_counts = np.asarray(class_counts).astype(int)
        # 记录类别总数
        self.num_classes = len(class_counts)
        # 设置类别名称，如果未提供则自动生成
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]
        
        # 保存分组相关的配置参数
        self.grouping = grouping          # 分组策略
        self.many_thresh = many_thresh    # 多数类绝对阈值
        self.few_thresh = few_thresh      # 少数类绝对阈值
        self.q_low = q_low                # 少数类分位数阈值
        self.q_high = q_high              # 多数类分位数阈值
        
        # 根据配置构建类别分组
        self._build_groups()

    def _build_groups(self):
       
        counts = self.class_counts
        
        if self.grouping == "absolute":
            # 绝对阈值模式：直接使用固定阈值划分
            many = np.where(counts >= self.many_thresh)[0]  # 样本数 >= 100 的类别
            few = np.where(counts <= self.few_thresh)[0]    # 样本数 <= 20 的类别
            
        elif self.grouping == "quantile":
            # 分位数模式：根据样本数分布动态划分
            lo = np.quantile(counts, self.q_low)   # 计算 1/3 分位数
            hi = np.quantile(counts, self.q_high)  # 计算 2/3 分位数
            many = np.where(counts >= hi)[0]       # 高于 2/3 分位数为多数类
            few = np.where(counts <= lo)[0]        # 低于 1/3 分位数为少数类
            
        else:
            # auto 模式：根据数据特征自动选择策略
            if counts.max() >= 100:
                # 如果最大样本数 >= 100，说明数据量较大，使用绝对阈值
                many = np.where(counts >= self.many_thresh)[0]
                few = np.where(counts <= self.few_thresh)[0]
            else:
                # 否则数据量较小，使用分位数划分更合理
                lo = np.quantile(counts, self.q_low)
                hi = np.quantile(counts, self.q_high)
                many = np.where(counts >= hi)[0]
                few = np.where(counts <= lo)[0]
        
        # 中等类 = 所有类别 - 多数类 - 少数类
        medium = np.setdiff1d(np.arange(self.num_classes), np.concatenate([many, few]))
        
        # 保存分组结果到实例属性
        self.majority_classes = many    # 多数类索引
        self.medium_classes = medium    # 中等类索引
        self.minority_classes = few     # 少数类索引
        
        # 打印分组信息，便于调试和验证
        print(f"Class grouping -> majority:{many.tolist()} | medium:{medium.tolist()} | minority:{few.tolist()}")

    def _group_indices(self) -> Dict[str, np.ndarray]:
        """
        获取各分组的类别索引字典
        
        Returns:
        --------
        Dict[str, np.ndarray]
            包含三个键值对的字典:
            - 'majority': 多数类索引数组
            - 'medium': 中等类索引数组
            - 'minority': 少数类索引数组
        """
        return {
            'majority': self.majority_classes,
            'medium': self.medium_classes,
            'minority': self.minority_classes
        }

    def _safe_topk(self, y_true: np.ndarray, prob: np.ndarray, k: int) -> Optional[float]:
      
        try:
            return float(top_k_accuracy_score(y_true, prob, k=k) * 100.0)
        except Exception:
            # 当类别数 < k 或其他异常时返回 None
            return None

    def analyze_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
       
        # 确保输入为 numpy 数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # ====================================================================
        # 第一部分：计算整体评估指标
        # ====================================================================
        
        # 1. 整体准确率 (Overall Accuracy)
        # 正确预测的样本数 / 总样本数
        overall_acc = float((y_true == y_pred).mean() * 100)
        
        # 2. 平衡准确率 (Balanced Accuracy)
        # 各类别召回率的算术平均，不受类别不平衡影响
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)

        # 3. 计算每个类别的精确率、召回率、F1分数和支持度
        # zero_division=0 表示当某类别无预测时，指标设为0
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 4. 宏平均指标 (Macro Average)
        # 先计算每个类别的指标，再取算术平均（每个类别权重相等）
        macro_precision = float(np.mean(precision) * 100)  # 宏平均精确率
        macro_recall = float(np.mean(recall) * 100)        # 宏平均召回率
        macro_f1 = float(np.mean(f1) * 100)                # 宏平均F1

        # 5. 微平均F1 (Micro F1)
        # 先汇总所有类别的TP/FP/FN，再计算F1（相当于加权平均，按样本数加权）
        _, _, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        micro_f1 = float(micro_f1 * 100)
        
        # 6. 几何平均召回率 (G-Mean)
        # 各类别召回率的几何平均，对类别不平衡更敏感
        # 如果任一类别召回率为0，G-Mean也为0
        # 使用 np.errstate 忽略 log(0) 的警告
        with np.errstate(divide='ignore'):
            # clip 防止 log(0) 产生 -inf
            gmean = float(np.exp(np.mean(np.log(np.clip(recall, 1e-12, 1.0)))) * 100)

        # ====================================================================
        # 第二部分：计算概率相关指标（需要 prob 参数）
        # ====================================================================
        
        top5 = macro_auroc = macro_auprc = None
        
        if prob is not None:
            # Top-5 准确率：真实类别在概率前5的比例
            if prob.ndim == 2 and prob.shape[1] >= 5:
                top5 = self._safe_topk(y_true, prob, k=5)
            
            # 当类别数 <= 200 时计算 AUROC 和 AUPRC（类别太多计算开销大）
            if prob.shape[1] <= 200:
                # Macro AUROC: One-vs-Rest 策略下的宏平均 ROC-AUC
                try:
                    macro_auroc = float(roc_auc_score(
                        y_true, prob, multi_class='ovr', average='macro'
                    ))
                except Exception:
                    macro_auroc = None
                
                # Macro AUPRC: 宏平均 PR曲线下面积
                try:
                    # 将标签转换为 one-hot 编码
                    y_true_ovr = np.eye(prob.shape[1])[y_true]
                    # 计算每个类别的 Average Precision
                    ap_list = [
                        average_precision_score(y_true_ovr[:, c], prob[:, c])
                        for c in range(prob.shape[1])
                    ]
                    macro_auprc = float(np.mean(ap_list))
                except Exception:
                    macro_auprc = None

        # ====================================================================
        # 第三部分：计算分组级别指标
        # ====================================================================
        
        group_metrics = {}
        groups = self._group_indices()  # 获取 majority/medium/minority 分组
        
        for gname, cls_idx in groups.items():
            # 如果该分组没有类别，跳过
            if len(cls_idx) == 0:
                group_metrics[gname] = {}
                continue
            
            # 创建掩码，筛选属于当前分组的样本
            mask = np.isin(y_true, cls_idx)
            
            # 如果该分组没有样本，跳过
            if mask.sum() == 0:
                group_metrics[gname] = {}
                continue
            
            # 计算该分组的各项指标
            gp_prec = float(np.mean(precision[cls_idx]) * 100)   # 分组平均精确率
            gp_rec = float(np.mean(recall[cls_idx]) * 100)       # 分组平均召回率
            gp_f1 = float(np.mean(f1[cls_idx]) * 100)            # 分组平均F1
            gp_s = int(support[cls_idx].sum())                    # 分组样本总数
            
            # 分组准确率：该分组内正确预测的比例
            gp_acc = float((y_true[mask] == y_pred[mask]).mean() * 100)
            
            # 分组平衡准确率
            gp_bal = float(balanced_accuracy_score(y_true[mask], y_pred[mask]) * 100)
            
            # 分组内最差类别的召回率（用于发现模型的薄弱类别）
            gp_worst = float(np.min(recall[cls_idx]) * 100)
            
            # 分组 Top-5 准确率
            gp_top5 = None
            if prob is not None and prob.shape[1] >= 5:
                try:
                    gp_top5 = float(top_k_accuracy_score(
                        y_true[mask], prob[mask], k=5
                    ) * 100.0)
                except Exception:
                    gp_top5 = None
            
            # 保存该分组的所有指标
            group_metrics[gname] = {
                'accuracy': gp_acc,
                'balanced_accuracy': gp_bal,
                'precision': gp_prec,
                'recall': gp_rec,
                'f1': gp_f1,
                'support': gp_s,
                'worst_class_recall': gp_worst,
                'top5': gp_top5
            }

        # ====================================================================
        # 第四部分：计算混淆矩阵和逐类别指标
        # ====================================================================
        
        # 计算混淆矩阵
        # cm[i, j] 表示真实类别为 i、预测类别为 j 的样本数
        cm = confusion_matrix(y_true, y_pred)
        
        # 构建逐类别指标字典
        class_metrics = {
            f"{i}": {
                'precision': float(precision[i] * 100),  # 精确率
                'recall': float(recall[i] * 100),        # 召回率
                'f1': float(f1[i] * 100),                # F1分数
                'support': int(support[i]),              # 测试集中该类别的样本数
                'frequency': int(self.class_counts[i])   # 训练集中该类别的样本数
            }
            for i in range(self.num_classes)
        }

        # ====================================================================
        # 第五部分：组装并返回完整的分析报告
        # ====================================================================
        
        return {
            # 整体评估指标
            'overall': {
                'accuracy': overall_acc,
                'balanced_accuracy': balanced_acc,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'gmean_recall': gmean,
                'top5': top5,
                'macro_auroc': macro_auroc,
                'macro_auprc': macro_auprc,
            },
            # 分组级别指标
            'group_wise': group_metrics,
            # 逐类别指标
            'per_class': class_metrics,
            # 原始混淆矩阵（计数）
            'confusion_matrix': cm.tolist(),
            # 归一化混淆矩阵（按行归一化，显示每个真实类别的预测分布）
            # 加 1e-12 防止除以零
            'confusion_matrix_normalized': (
                cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
            ).tolist(),
            # 最差类别召回率（用于衡量模型的最差表现）
            'worst_class_recall': float(np.min(recall) * 100) if recall.size > 0 else 0.0,
            # 分组策略的元信息，便于结果复现
            'grouping_meta': {
                'strategy': self.grouping,
                'many_thresh': int(self.many_thresh),
                'few_thresh': int(self.few_thresh),
                'q_low': float(self.q_low),
                'q_high': float(self.q_high)
            }
        }

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str,
        normalize: bool = False,
        figsize: Tuple[int, int] = (10, 8)
    ):
        # 检查绘图库是否可用
        if not HAS_PLOTTING:
            print("Plotting libs unavailable, skip CM plot.")
            return
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 根据 normalize 参数决定显示内容
        if normalize:
            # 归一化：按行求和后除以行和
            cm_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
            fmt = '.2f'  # 显示两位小数
            title = 'Normalized Confusion Matrix'
        else:
            cm_plot = cm
            fmt = 'd'  # 显示整数
            title = 'Confusion Matrix'
        
        # 使用 seaborn 绘制热力图
        # annot=True 在每个格子中显示数值
        # cmap='Blues' 使用蓝色色阶
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues')
        
        # 设置标题和轴标签
        plt.title(title)
        plt.xlabel('Predicted')  # 横轴：预测类别
        plt.ylabel('True')       # 纵轴：真实类别
        
        # 自动调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存
        
        print(f"Confusion matrix saved to: {save_path}")
