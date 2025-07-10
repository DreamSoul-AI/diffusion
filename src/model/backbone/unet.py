from .layers import *


class UNet(nn.Module):
    def __init__(self, data_size, hidden_size, time_embedding_size, cond_embedding_size):
        super().__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.time_embedding_size = time_embedding_size
        self.cond_embedding_size = cond_embedding_size
        c = hidden_size  # The base channel count

        self.net = nn.Sequential(  # 32x32
            ResConvBlock(self.data_size[0] + time_embedding_size + cond_embedding_size, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        SelfAttention2d(c * 4, c * 4 // 64),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    SelfAttention2d(c * 2, c * 2 // 64),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, self.data_size[0], is_last=True),
        )

    def forward(self, x, time_embedding=None, cond_embedding=None):
        if time_embedding is not None:
            time_embedding = expand_shape(time_embedding, x.size(), 'base')
            x = torch.cat([x, time_embedding], dim=1)
        if cond_embedding is not None:
            cond_embedding = expand_shape(cond_embedding, x.size(), 'base')
            x = torch.cat([x, cond_embedding], dim=1)
        x = self.net(x)
        return x




def unet(cfg):
    data_size = cfg['data_size']
    hidden_size = cfg['unet']['hidden_size']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = UNet(data_size, hidden_size, time_embedding_size, cond_embedding_size)
    return model
