# @Time : 2023/4/26 16:02 
# @Author : Li Jiaqi
# @Description :
import torch.nn as nn
import torch


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, froze_encoder=False, vit_encoder=True):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vit_encoder = vit_encoder
        self.h_patch_num = self.decoder.img_size[0] // self.decoder.patch_size
        self.w_patch_num = self.decoder.img_size[1] // self.decoder.patch_size

        # frozen the pre-trained model
        if froze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.vit_encoder:
            # combine the clstoken with patchtokens
            x_temp = self.encoder.forward_features(x)
            x = torch.zeros((x_temp["x_norm_patchtokens"].shape[0], x_temp["x_norm_patchtokens"].shape[1] + 1,
                             x_temp["x_norm_patchtokens"].shape[2])).cuda()
            x[:, 0] = x_temp["x_norm_clstoken"]
            x[:, 1:] = x_temp["x_norm_patchtokens"]
        else:
            x = self.encoder(x)
        output = self.decoder(x)
        return output, x  # B,C,H,W

    def flatten(self, x):
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = x.reshape(-1, self.h_patch_num * self.w_patch_num,
                      self.decoder.out_chans * self.decoder.patch_size ** 2)  # B,P,C*Patch_size
        return x

    def patch_loss(self, pred, target):
        pred = self.flatten(pred)
        target = self.flatten(target)
        # loss = (pred_feature - original_feature) ** 2 # l2-loss
        loss_abs = torch.abs(pred - target)  # l1-loss
        loss_mean = loss_abs.mean(dim=-1) / (self.h_patch_num * self.w_patch_num)
        loss = loss_mean.sum()
        if torch.isnan(loss):
            print('nan!')
        return loss

    def forward_encoder(self, x):
        # combine the clstoken with patchtokens
        x_temp = self.encoder.forward_features(x)
        x = torch.zeros((x_temp["x_norm_patchtokens"].shape[0], x_temp["x_norm_patchtokens"].shape[1] + 1,
                         x_temp["x_norm_patchtokens"].shape[2])).cuda()
        x[:, 0] = x_temp["x_norm_clstoken"]
        x[:, 1:] = x_temp["x_norm_patchtokens"]
        return x

    def cycle_loss(self, pred, original_feature):
        pred_feature = self.forward_encoder(pred)
        # loss = (pred_feature - original_feature) ** 2 # l2-loss
        loss = torch.abs(pred_feature - original_feature)  # l1-loss
        loss = loss.mean(dim=-1)
        loss = loss.sum()
        return loss
