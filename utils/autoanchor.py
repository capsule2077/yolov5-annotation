# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(f'{PREFIX}ERROR')
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # 检查anchor是否适合数据，如果不适合则重新计算
    # Check anchor fit to data, recompute if necessary
    # 把最后的检测层取出来(也就是最后预测的三个255通道的卷积层)
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()

    # dataset.shapes记录了训练集上所有图像大小，以coco128数据集为例，是一个[128, 2]的numpy
    # array。这句找到128张图像上宽高最大的数值，将dataset.shapes / 最大值
    # 归一化到0到1，按照比例所放到最大640边长。
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)

    # augment scale表示对每张图像的缩放比例，这里是一个[128, 1]的numpy array，每个值在0.9到1.1之间
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale

    # dataset.labels dataset.labels其中格式为[类别，x, y , w, h]，xywh均已经归一化了，最终得到缩放之后标签的宽和高，
    # shape为[929，2]的tensor, coco128训练集上共有929个标注框，由此得到wh一个轻微扰动后在训练集上所有标注框的宽高的集合
    # l[:, 3:5]选取标签中的w和h，乘以scale，得到扰动后的标签的w和h
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    # 闭包
    # metric其实输入是anchors[9x2]和第4步中wh[929x2]
    def metric(k):  # compute metric
        # 计算计算得到929个标注框的宽高与9个anchor宽高取比值，宽比宽，高比高。
        # 这时候标注框有比anchor大的有比他小的，且尺度从零点几到几百都有可能，很难设定阈值。
        r = wh[:, None] / k[None]

        # r.shape = [929, 9, 2]
        # torch.min(r, 1 / r)取r和1/r中的最小值，这样就保证了r中的值都是小于1的，也就是说，如果标注框的宽高比anchor大，那么
        # 1/r中的值就会小于r中的值，这样取最小值就是1/r中的值，反之就是r中的值。
        # torch.min(r, 1/r)逐元素比较选取两个tensor的最小值
        # 第二个min在第2维度上也就是列方向上选取最小值，min()返回一个元组，第一个元素是最小值，第二个元素是最小值的索引
        # 把宽高比中的最小值取出来
        # x.shape = [929, 9]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # 选出9个anchor中最大的那个也就是最匹配GT的那个
        best = x.max(1)[0]  # best_x

        # aat表示在训练集上平均有几个anchor超过阈值，所有anchor都参与计算。例如使用coco128 / yolov5s配置文件计算为4.26695，
        # 表示平均每个标签可以匹配4.26个anchor，这个结果也是很不错的。
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold

        # bpr表示最佳的recall，也就是说，如果所有的标签都能匹配到最佳的anchor，那么recall就是1，如果有一半的标签能匹配到最佳的anchor，
        # 那么recall就是0.5，这个值越大越好，这个值越大，说明anchor越适合数据集。
        # 也就是每个GT匹配一个最好的Anchor判断有多少Anchor能够超过阈值
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        LOGGER.info(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')
        na = m.anchors.numel() // 2  # number of anchors
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(s)


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
