import random

import numpy as np
import torch

import os
import pandas as pd

from data.preprocessing import load_data, transfer_labels, k_fold
from models.loss import cross_entropy, reconstruction_loss
from sklearn.metrics import accuracy_score

import csv
import matplotlib.pyplot as plt
from pathlib import Path

# 新增：设置matplotlib中文支持（避免中文标题乱码，可选）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac用
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux用
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)
    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    elif args.loss == 'reconstruction':
        return reconstruction_loss()


def get_all_datasets(data, target):
    return k_fold(data, target)


def evaluate_model(val_loader, model, loss):
    val_loss = 0
    val_pred_labels = []
    real_labels = []

    sum_len = 0
    for data, target in val_loader:
        with torch.no_grad():
            val_pred, _ = model(data)
            val_loss = val_loss + loss(val_pred, target).item()
            sum_len = sum_len + len(target)
            val_pred_labels.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
            real_labels.append(target.cpu().numpy())

    val_pred_labels = np.concatenate(val_pred_labels)
    real_labels = np.concatenate(real_labels)

    return val_loss / sum_len, accuracy_score(real_labels, val_pred_labels)


def save_cls_result(args, mean_accu, train_time, best_accu):
    save_path = os.path.join(args.save_dir, '', args.save_csv_name + '_cls.csv')
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path, index_col=0)
    else:
        result_form = pd.DataFrame(columns=['dataset_name', 'mean_accu', 'best_accu', 'train_time'])

    result_form = pd.concat([result_form,
                             pd.DataFrame([{
                                 'dataset_name': args.dataset,
                                 'mean_accu': '%.4f' % mean_accu,
                                 'best_accu': '%.4f' % best_accu,
                                 'train_time': '%.4f' % train_time
                             }])], ignore_index=True)

    result_form.to_csv(save_path, index=True, index_label="id")


def plot_train_curves(
    dataset_name: str,
    epochs: list,
    train_accs: list,
    val_accs: list,
    train_losses: list,
    val_losses: list,
    save_dir: str = "./result/plots"
):
    """
    绘制并保存单个数据集的训练曲线（Accuracy + Loss 分开绘制）
    Args:
        dataset_name: 数据集名称（如CBF、Coffee）
        epochs: 训练轮数列表（横坐标，如[1,2,...,500]）
        train_accs: 每轮训练集准确率
        val_accs: 每轮验证集准确率
        train_losses: 每轮训练集损失
        val_losses: 每轮验证集损失
        save_dir: 图片保存根目录（默认在result/plots下）
    """
    # 1. 创建保存文件夹（不存在则自动创建）
    dataset_plot_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_plot_dir, exist_ok=True)  # exist_ok=True避免重复创建报错

    # 2. 创建画布（2个子图：上为Accuracy，下为Loss）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

    # ---------------------- 子图1：Accuracy曲线 ----------------------
    ax1.plot(epochs, train_accs, label=f"Train Sets Accuracy", color="#2E86AB", linewidth=2)
    ax1.plot(epochs, val_accs, label=f"Validation Sets Accuracy", color="#A23B72", linewidth=2, linestyle="--")
    # 标记最高准确率（验证集）
    best_val_acc = max(val_accs)
    best_epoch = epochs[val_accs.index(best_val_acc)]
    ax1.scatter(best_epoch, best_val_acc, color="red", s=50, zorder=5)  # 红色圆点标记
    ax1.annotate(
        f"Best Accuracy: {best_val_acc:.2%}\nEpoch: {best_epoch}",
        xy=(best_epoch, best_val_acc),
        xytext=(best_epoch+10, best_val_acc-5),  # 文本位置偏移
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
    )
    # 设置子图1样式
    ax1.set_title(f"{dataset_name} Accuracy Curve", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)  # 准确率范围固定在0-1.05，更直观

    # ---------------------- 子图2：Loss曲线 ----------------------
    ax2.plot(epochs, train_losses, label=f"Train Sets Loss", color="#F18F01", linewidth=2)
    ax2.plot(epochs, val_losses, label=f"Validation Sets Loss", color="#C73E1D", linewidth=2, linestyle="--")
    # 标记最低损失（验证集）
    best_val_loss = min(val_losses)
    best_loss_epoch = epochs[val_losses.index(best_val_loss)]
    ax2.scatter(best_loss_epoch, best_val_loss, color="blue", s=50, zorder=5)
    ax2.annotate(
        f"Lowest Loss: {best_val_loss:.4f}\nEpoch: {best_loss_epoch}",
        xy=(best_loss_epoch, best_val_loss),
        xytext=(best_loss_epoch+10, best_val_loss+0.1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2")
    )
    # 设置子图2样式
    ax2.set_title(f"{dataset_name} Loss Curve", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 保存图片（PNG格式，高清）
    save_path = os.path.join(dataset_plot_dir, f"{dataset_name}_train_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭画布，避免内存泄漏
    print(f"✅ Train Curves has been successfully saved to：{save_path}")

