import matplotlib.pyplot as plt
import numpy as np

def plot_combined_metrics_aligned():
    # ================= 数据准备 =================
    labels = ['Easy Set', 'Hard Set']
    
    # AUC 数据
    auc_baseline = [0.8620, 0.8605]
    auc_af2 = [0.8542, 0.8871]
    auc_extended = [0.8834, 0.8819]  # 第三组数据

    # AUPR 数据
    aupr_baseline = [0.4914, 0.2805]
    aupr_af2 = [0.3735, 0.3286]
    aupr_extended = [0.5151, 0.3279] # 第三组数据

    # ================= 绘图设置 =================
    x = np.arange(len(labels))
    width = 0.25  # 宽度调小以容纳三组柱子

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    color_base = '#1f77b4' 
    color_af2 = '#d62728'   
    color_ext = '#2ca02c'   

    # ================= 左图 (AUC) =================
    rects1_auc = ax1.bar(x - width, auc_baseline, width, label='Baseline', color=color_base, alpha=0.9)
    rects2_auc = ax1.bar(x, auc_af2, width, label='AF2', color=color_af2, alpha=0.9)
    rects3_auc = ax1.bar(x + width, auc_extended, width, label='Extended', color=color_ext, alpha=0.9)

    ax1.set_ylabel('AUC Score', fontsize=14)
    ax1.set_title('ROC-AUC', fontsize=18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=16)
    
    # [核心修改]：左图 Y 轴范围从 0.80 开始，上限留余地到 0.92
    ax1.set_ylim(0.80, 0.92) 
    # [核心修改]：设置固定的刻度间隔为 0.02
    ax1.set_yticks(np.arange(0.80, 0.93, 0.02))
    
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)

    # ================= 右图 (AUPR) =================
    rects1_aupr = ax2.bar(x - width, aupr_baseline, width, label='Baseline', color=color_base, alpha=0.9)
    rects2_aupr = ax2.bar(x, aupr_af2, width, label='AF2', color=color_af2, alpha=0.9)
    rects3_aupr = ax2.bar(x + width, aupr_extended, width, label='Extended', color=color_ext, alpha=0.9)

    ax2.set_ylabel('AUPR Score', fontsize=14)
    ax2.set_title('AUPR', fontsize=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=16)
    
    # [核心修改]：右图 Y 轴范围从 0.20 开始，上限留余地到 0.55
    ax2.set_ylim(0.20, 0.55) 
    # [核心修改]：设置固定的刻度间隔为 0.05
    ax2.set_yticks(np.arange(0.20, 0.56, 0.05))
    
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)

    # ================= 统一图例 =================
    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='upper center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, 1.08))

    # ================= 自动标注数值 =================
    def autolabel(ax, rects):
        for rect in rects:
            height = rect.get_height()
            # 只有当柱子高度大于 Y 轴底端时才标注，避免文字超出图表下方
            if height > ax.get_ylim()[0]:
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)

    # 调用标注函数
    autolabel(ax1, rects1_auc)
    autolabel(ax1, rects2_auc)
    autolabel(ax1, rects3_auc)
    
    autolabel(ax2, rects1_aupr)
    autolabel(ax2, rects2_aupr)
    autolabel(ax2, rects3_aupr)
    
    plt.tight_layout()
    
    # ================= 保存为 SVG =================
    save_path = 'combined_metrics_comparison_aligned.svg'
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_combined_metrics_aligned()