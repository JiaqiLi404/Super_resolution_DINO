# @Time : 2023/4/22 18:00
# @Author : Li Jiaqi
# @Description :
import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import numpy as np
import cv2
from models.ViT_Decoder import Decoder


def sift_algo(img1_tensor, img2_tensor):
    img1_cv = img1_tensor.permute(1, 2, 0)
    img1_cv = img1_cv * 255
    img1_cv = img1_cv.numpy()
    img1_cv = img1_cv.astype(np.uint8).copy()
    img2_cv = img2_tensor.permute(1, 2, 0)
    img2_cv = img2_cv * 255
    img2_cv = img2_cv.numpy()
    img2_cv = img2_cv.astype(np.uint8).copy()
    img = np.append(img1_cv, img2_cv, axis=1)

    img1_grey = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2GRAY)
    img2_grey = cv2.cvtColor(img2_cv, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None)
    keyPointsLeft, describesLeft = sift.detectAndCompute(img1_grey, None)
    keyPointsRight, describesRight = sift.detectAndCompute(img2_grey, None)
    # K-D tree建立索引方式的常量参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # checks指定索引树要被遍历的次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_1 = flann.knnMatch(describesLeft, describesRight, k=2)  # 进行匹配搜索，参数k为返回的匹配点对数量
    # 把保留的匹配点放入good列表
    good1 = []
    T = 0.85  # 阈值
    # 筛选特征点
    for i, (m, n) in enumerate(matches_1):
        if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
            good1.append(m)
        #  双向交叉检查方法
    matches_2 = flann.knnMatch(describesRight, describesLeft, k=2)  # 进行匹配搜索
    # 把保留的匹配点放入good2列表
    good2 = []
    for (m, n) in matches_2:
        if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
            good2.append(m)
    match_features = []  # 存放最终的匹配点
    for i in good1:
        for j in good2:
            if (i.trainIdx == j.queryIdx) & (i.queryIdx == j.trainIdx):
                match_features.append(i)
    src_pts = [keyPointsLeft[m.queryIdx].pt for m in match_features]
    dst_pts = [keyPointsRight[m.trainIdx].pt for m in match_features]
    for i in range(min(len(src_pts), 10)):
        img = cv2.line(img, (int(src_pts[i][0]), int(src_pts[i][1])),
                       (int(dst_pts[i][0] + img1_cv.shape[1]), int(dst_pts[i][1])), (255, 255, 0), 2)
    cv2.imshow('sift', img)
    cv2.waitKey(0)
    return img


def patch2piex(patch, h, w, patch_num):
    patch_num_per_direction = patch_num ** 0.5
    patch_w = w / patch_num_per_direction
    patch_h = h / patch_num_per_direction
    patch_h_id = patch // patch_num_per_direction
    patch_w_id = patch % patch_num_per_direction
    return (int(0.5 * patch_h + patch_h_id * patch_h), int(0.5 * patch_w + patch_w_id * patch_w))


def draw_patch_corresponding_lines(img1_tensor, img1_patches, img2_tensor, img2_patches, patch_num):
    img1_cv = img1_tensor.permute(1, 2, 0)
    img1_cv = img1_cv * 255
    img1_cv = img1_cv.numpy()
    img1_cv = img1_cv.astype(np.uint8).copy()
    img2_cv = img2_tensor.permute(1, 2, 0)
    img2_cv = img2_cv * 255
    img2_cv = img2_cv.numpy()
    img2_cv = img2_cv.astype(np.uint8).copy()
    img = np.append(img1_cv, img2_cv, axis=1)
    # img = np.ascontiguousarray(img, dtype=np.uint8)
    for patch_i in range(len(img1_patches)):
        piex1 = patch2piex(img1_patches[patch_i], img1_cv.shape[0], img1_cv.shape[1], patch_num)
        piex2 = patch2piex(img2_patches[patch_i], img2_cv.shape[0], img2_cv.shape[1], patch_num)
        img = cv2.line(img, (piex1[1], piex1[0]), (img1_cv.shape[1] + piex2[1], piex2[0]), (255, 255, 0), 2)

    cv2.imshow('dino', img)
    # cv2.waitKey(0)
    return img


if __name__ == '__main__':
    vis = visdom.Visdom(env='plot1')
    data = "../Datas/AreialImage/ArchaeologicalSitesDetection/georgia_cleaned_all"
    dino_encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dino_encoder_model.cuda()
    # frozen the pre-trained model
    for param in dino_encoder_model.parameters():
        param.requires_grad = False
    dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig)
    print(len(dataLoader))
    for bing_image, book_image in dataLoader:
        # extract the feature of both image
        bing_image_cuda = bing_image.cuda()
        bing_patch_feature = dino_encoder_model.forward_features(bing_image_cuda)['x_norm_patchtokens']
        bing_patch_feature = bing_patch_feature.detach().cpu()
        book_image_cuda = book_image.cuda()
        book_patch_feature_cuda = dino_encoder_model.forward_features(book_image_cuda)
        book_image_feature_cuda=book_patch_feature_cuda['x_norm_clstoken']
        book_patch_feature_cuda =book_patch_feature_cuda['x_norm_patchtokens']
        book_patch_feature = book_patch_feature_cuda.detach().cpu()
        # get the similarity matrix and find the corresponding patches
        patch_num = int(bing_patch_feature.shape[1])
        batch_num = int(bing_patch_feature.shape[0])
        similarity = torch.zeros(size=(batch_num, patch_num, patch_num), device='cpu')
        patch_pairs = []  # [batch,pairs,[sim,patch1,patch2]]
        for batch_i in range(batch_num):
            patch_pairs.append([])
            for patch_i in range(patch_num):
                similarity[batch_i, patch_i, :] = torch.cosine_similarity(bing_patch_feature[batch_i, patch_i, :],
                                                                          book_patch_feature[batch_i, :, :], dim=-1)
                max_id = torch.argmax(similarity[batch_i, patch_i, :], dim=-1)
                patch_pairs[-1].append([float(similarity[batch_i, patch_i, max_id]), patch_i, int(max_id)])
            patch_pairs[-1] = sorted(patch_pairs[-1], key=lambda x: x[0], reverse=True)
            patch_pairs[-1] = patch_pairs[-1][:10]

        # draw the corresponding patches
        draw_patch_corresponding_lines(bing_image[0], [x[1] for x in patch_pairs[0]], book_image[0],
                                       [x[2] for x in patch_pairs[0]], patch_num)
        sift_algo(bing_image[0], book_image[0])

        # train the AutoEncoder(decoder)
        decoder = Decoder(1024).cuda()
        book_image_recovered = decoder(book_patch_feature_cuda)

        # show the image to Visdom
        bing_image_numpy = bing_image.numpy()
        bing_image_numpy = bing_image_numpy[0]
        vis.image(bing_image_numpy)
        book_image_numpy = book_image.numpy()
        book_image_numpy = book_image_numpy[0]
        vis.image(book_image_numpy)

        break
