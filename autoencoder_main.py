# @Time : 2023/4/22 18:00
# @Author : Li Jiaqi
# @Description :
import os.path

import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import numpy as np
import cv2
from models.ViT_Decoder import Decoder
from models.ViT_AutoEncoder import AutoEncoder

# python -m visdom.server

if __name__ == '__main__':
    vis = visdom.Visdom(env='plot1')
    dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig)
    print('batch amount: ', len(dataLoader))

    data = "../Datas/AreialImage/ArchaeologicalSitesDetection/georgia_cleaned_all"
    # dino_encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dino_encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino_encoder_model.cuda()
    decoder = Decoder(img_size=(config.ModelConfig['imgh'], config.ModelConfig['imgw']),
                      patch_size=dino_encoder_model.patch_size, depth=dino_encoder_model.n_blocks,
                      embed_dim=dino_encoder_model.embed_dim, num_heads=dino_encoder_model.num_heads).cuda()
    ae_model = AutoEncoder(dino_encoder_model, decoder)
    if os.path.exists(os.path.join('checkpoints', 'pretrain.pth')):
        ae_model.load_state_dict(torch.load(os.path.join('checkpoints', 'pretrain.pth')))
        print('pretrained model loaded')

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, ae_model.parameters()),
                                 lr=config.ModelConfig['lr'], weight_decay=config.ModelConfig['weight_decay'],
                                 betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.ModelConfig['scheduler'])
    loss_function = torch.nn.L1Loss()
    for epoch_i in range(config.ModelConfig['epoch_num']):
        epoch_loss = []
        for bing_image, book_image in dataLoader:
            ae_model.train()
            # cuda tensors
            bing_image_cuda = bing_image.cuda()
            book_image_cuda = book_image.cuda()

            # train the AutoEncoder(decoder)
            recovery_image_cuda, bing_feature_cuda = ae_model(bing_image_cuda)
            recovery_image = recovery_image_cuda.detach().cpu()
            resolution_loss = ae_model.patch_loss(recovery_image_cuda, bing_image_cuda)

            # CycleGAN
            cycle_loss = 0
            # cycle_loss = ae_model.cycle_loss(recovery_image_cuda, bing_feature_cuda)

            # loss = 0.7 * cycle_loss + 0.3 * resolution_loss
            loss = resolution_loss
            optimizer.zero_grad()
            if not torch.isnan(loss): loss.backward()
            if len(epoch_loss) % 5 == 0:
                print('loss:{0:.6f} resoLoss:{1:.4f} cycleLoss:{2:.4f}'.format(float(loss), float(resolution_loss),
                                                                               float(cycle_loss)))
            epoch_loss.append(float(loss))
            # torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=20, norm_type=2)
            torch.nn.utils.clip_grad_value_(ae_model.parameters(), clip_value=1.2)
            optimizer.step()

            # show the image to Visdom
            if len(epoch_loss) % 20 == 0:
                bing_image_numpy = bing_image.numpy()
                bing_image_numpy = bing_image_numpy[0]
                vis.image(bing_image_numpy)
                book_image_numpy = book_image.numpy()
                book_image_numpy = book_image_numpy[0]
                vis.image(book_image_numpy)
                recovery_image_numpy = recovery_image.numpy()
                recovery_image_numpy = recovery_image_numpy[0]
                vis.image(recovery_image_numpy)
        # save model
        if epoch_i % 5 == 0:
            torch.save(ae_model.state_dict(),
                       os.path.join('checkpoints', 'epoch {0} loss {1:.3f}.pth'.format(epoch_i, sum(epoch_loss))))
        print('--------epoch {0} loss: {1:.6f}'.format(epoch_i, sum(epoch_loss)))
        scheduler.step()
        print()
