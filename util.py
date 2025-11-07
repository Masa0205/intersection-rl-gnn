import matplotlib.pyplot as plt
import os

def r_graph(r_total, timestamp):
    
    # === 学習終了後 ===
    plt.figure(figsize=(10, 6))

    for inter, rewards in r_total.items():
        plt.plot(rewards, label=f'{inter}')

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward per Step", fontsize=12)
    plt.title("Reward Progress per Intersection", fontsize=14)
    plt.legend(title="Intersection ID")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # 保存先ディレクトリを確認・作成
    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)

    # 画像として保存（PNG形式）
    save_path = os.path.join(save_dir, f"reward_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300)

    plt.close()  

    print(f"グラフを保存しました: {save_path}")

def loss_graph(loss_total, timestamp):
    
    # === 学習終了後 ===
    plt.figure(figsize=(10, 6))

    for inter, rewards in loss_total.items():
        plt.plot(rewards, label=f'{inter}')

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average loss per Step", fontsize=12)
    plt.title("Reward Progress per Intersection", fontsize=14)
    plt.legend(title="Intersection ID")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # 保存先ディレクトリを確認・作成
    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)

    # 画像として保存（PNG形式）
    save_path = os.path.join(save_dir, f"loss_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300)

    plt.close()  

    print(f"グラフを保存しました: {save_path}")

def t_graph(t_totals, timestamp):
    
    # === 学習終了後 ===
    plt.figure(figsize=(10, 6))

    for t in t_totals:
        plt.plot(t, label=f'time')

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average time per 10 eps", fontsize=12)
    plt.title("time av Progress per Intersection", fontsize=14)
    plt.legend(title="time")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # 保存先ディレクトリを確認・作成
    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)

    # 画像として保存（PNG形式）
    save_path = os.path.join(save_dir, f"time_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300)

    plt.close()  

    print(f"グラフを保存しました: {save_path}")