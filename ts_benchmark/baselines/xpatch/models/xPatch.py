import torch
import torch.nn as nn
from einops import rearrange
import math

from ..layers.decomp import DECOMP
from ..layers.network_time import Network
# from layers.network_mlp import NetworkMLP # For ablation study with MLP-only stream
# from layers.network_cnn import NetworkCNN # For ablation study with CNN-only stream
from ..layers.revin import RevIN
from ts_benchmark.baselines.xpatch.layers.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer

class xPatchModel(nn.Module):
    def __init__(self, configs):
        super(xPatchModel, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch,configs.d_model)
        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream


        # ======================== 生成通道相关性聚类 ========================
        # TODO 超参数调试
        # 初始化掩码矩阵生成器
        self.mask_generator = Mahalanobis_mask(configs.seq_len)
        # 初始化时间-通道相关性特征融合模块
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 最终输出层
        self.output_layer = nn.Sequential(nn.Linear(configs.d_model, configs.pred_len), nn.Dropout(configs.fc_dropout))


    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            output = self.net(x, x)
            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        else:
            # 趋势-季节分解
            seasonal_init, trend_init = self.decomp(x)
            time_feature = self.net(seasonal_init, trend_init) # [batch_size,var_nums,d_model]
            # 生成通道相关性掩码矩阵
            changed_input = rearrange(x, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)  # [batch_size,1,var_nums,var_nums]
            # 时间-通道相关系 特征融合
            channel_group_feature, attention = self.Channel_transformer(x=time_feature, attn_mask=channel_mask)
            output = self.output_layer(channel_group_feature)

        # Denormalization
        if self.revin:
            output = rearrange(output, 'b n d -> b d n')  # [batch_size,pre_len,var_nums]
            output = self.revin_layer(output, 'denorm')

        return output