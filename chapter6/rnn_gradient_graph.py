#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def main():
    N = 2  # ミニバッチサイズ
    H = 3  # 隠れ状態ベクトルの次元数
    T = 20  # 時系列データの長さ

    dh = np.ones((N, H))
    np.random.seed(3)  # 再現性のため乱数のシードを固定
    # befor
    # Wh = np.random.randn(H, H)
    # after
    Wh = np.random.randn(H, H) * 0.5

    norm_list = []
    for t in range(T):
        dh = np.dot(dh, Wh.T)
        norm = np.sqrt(np.sum(dh**2)) / N
        norm_list.append(norm)

    print(norm_list)

    # グラフの描画
    plt.plot(np.arange(len(norm_list)), norm_list)
    plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    plt.xlabel('time step')
    plt.ylabel('norm')
    plt.show()

if __name__ == '__main__':
    main()
