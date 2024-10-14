import torch
from .subsample import subsample


def loss_func(model, origin, loss_function=None):
    """
    柔軟な損失計算関数

    :param model: ノイズ除去モデル
    :param origin: 元の音声データ
    :param loss_function: 任意の損失関数（デフォルトはMSELoss）
    :return: 計算された損失値
    """
    if loss_function is None:
        loss_function = torch.nn.MSELoss()

    # originを2つにサブサンプル
    g1, g2 = subsample(origin)

    # モデルからの予測と残差の計算
    pred1 = g1 - model(g1)
    pred2 = g2 - model(g2)

    # 損失の計算（Resコンポーネント）
    loss_res = 0.5 * (loss_function(g1, pred2) + loss_function(g2, pred1))

    # Denoising後の計算
    denoised = origin - model(origin)
    dg1, dg2 = subsample(denoised)

    # 損失の計算（Consコンポーネント）
    loss_cons = 0.5 * (loss_function(pred1, dg1) + loss_function(pred2, dg2))

    # 最終的な損失の合計
    loss = loss_res + loss_cons

    return loss
