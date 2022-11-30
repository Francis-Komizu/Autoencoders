import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from modules import *


class ContentEncoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, freq):
        super(ContentEncoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        # convolution layers
        convolutions = []
        """
        The concatenated features are fed into three 5 × 1 convolutional layers, 
        each followed by batch normalization and ReLU activation. The number of channels is 512.
        """
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80 + dim_emb if i == 0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2, dilation=1, w_gain_init='relu'),  # padding=2 keeps the sequence length
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # LSTM layers
        self.lstm = nn.LSTM(512, dim_neck, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x, spk_emb):
        # concatenate 80-dim mel-spectrogram and 256-dim speaker embedding
        # input spk_emb: 2-dim LongTensor [1, n], where n is he speaker id
        # input mel: [B,  , N]
        spk_emb = spk_emb.unsqueeze(-1).expand(x.size(0), -1, x.size(-1))  # expand spk_emb
        x = torch.cat((x, spk_emb), dim=1)  # [B, C, N], where C=80+256=336
        for conv in self.convolutions:  # convolutions along time-axis
            x = conv(x)  # [B, C, N]
        x = x.transpose(1, 2)  # [B, N, C]
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # bidirectional LSTM layers
        out_forward = outputs[:, :, :self.dim_neck]  # [B, N, dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]  # [B, N, dim_neck]

        """
        As a key step of constructing the information bottleneck, 
        both the forward and backward outputs of the bidirectional LSTM are down-sampled by 32. 
        The down-sampling is performed differently for the forward and backward paths. 
        For the forward output, the time steps {0, 32, 64, · · · } are kept;
        for the backward output, the time steps {31, 63, 95, · · · } are kept.
        
        The down-sampling can be regarded as dimension reduction along the temporal axis, 
        which, together with the dimension reduction along the channel axis, constructs the information bottleneck.
        """
        # down-sampling along time-axis
        codes = []
        for i in range(0, outputs.size(1), self.freq):  # freq: down-sampling factor
            code_forward = out_forward[:, i + self.freq - 1, :]  # NOTE: reverse?
            code_backward = out_backward[:, i, :]
            code = torch.cat((code_forward, code_backward), dim=-1)
            codes.append(code)
        return codes


class SpeakerEncoder(nn.Module):
    def __init__(self, n_speakers, spk_emb):
        """
        :param n_speakers: number of speakers (must > 1)
        """
        super(SpeakerEncoder, self).__init__()
        if n_speakers < 2:
            raise ValueError('Number of speakers must > 0.')
        self.emb = nn.Embedding(n_speakers, spk_emb)
        nn.init.normal_(self.emb.weight, mean=0.0, std=spk_emb ** -0.5)

    def forward(self, x):
        return self.emb(x)


class Decoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(2 * dim_neck + dim_emb, dim_pre, num_layers=1, batch_first=True)

        """
        First, the content and speaker embeddings are both upsampled by copying 
        to restore to the original temporal resolution.
        
        U→(:, t) = C1→(:, t / 32)
        U←(:, t) = C1←(:, t / 32)
        
        The underlying intuition is that each embedding at each time step should contain both past and future 
        information. For the speaker embedding, simply copy the vector T times. 
        
        Then, the upsampled embeddings are concatenated and fed into three 5×1 convolutional layers with 512 channels, 
        each followed by batch normalization and ReLU, and then three LSTM layers with cell dimension 1024. 
        The outputs of the layer are projected to dimension 80 with a 1 × 1 convolutional layer.
        """
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2, dilation=1, w_gain_init='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(dim_pre, 1024, num_layers=3, batch_first=True)

        self.linear_proj = LinearNorm(1024, 80)

    def forward(self, x):   # [B, N, C]
        x, _ = self.lstm1(x)  # NOTE: why lstm?
        x = x.transpose(1, 2)   # [B, dim_pre, N]
        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)   # [B, N, dim_pre]
        y, _ = self.lstm2(x)   # [B, N, 1024]

        out = self.linear_proj(y)   # [B, N, 80]

        return out


class PostNet(nn.Module):
    """
    In order to construct the fine details of the spectrogram better on top of the initial estimate,
    we introduce a post network after the initial estimate.
    """

    def __init__(self):
        super(PostNet, self).__init__()
        """
        The post network consists of five 5×1 convolutional layers, 
        where batch normalization and hyperbolic tangent are applied to the first four layers. 
        The channel dimension for the first four layers is 512, 
        and goes down to 80 in the final layer.
        """
        convolutions = []
        convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2, dilation=1, w_gain_init='tanh'),
                nn.BatchNorm1d(512),
                nn.Tanh()))

        for i in range(1, 5 - 1):
            convolutions.append(
                nn.Sequential(
                    ConvNorm(512, 512,
                             kernel_size=5, stride=1,
                             padding=2, dilation=1, w_gain_init='tanh'),
                    nn.BatchNorm1d(512),
                    nn.Tanh()))

        convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2, dilation=1, w_gain_init='linear'),
                nn.BatchNorm1d(80),
                nn.Tanh()))

        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):   # [B, N, 80]
        x = x.transpose(1, 2)   # [B, 80, N]
        for conv in self.convolutions:
            x = conv(x)

        return x    # [B, 80, N]


class Generator(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, n_speakers):
        """
        :param dim_neck: dimension of encoder information bottleneck
        :param dim_emb: dimension of speaker embedding
        :param dim_pre: dimension of decoder pre-net
        :param freq: dimension of encoder down-sampling factor
        """
        super(Generator, self).__init__()

        self.content_encoder = ContentEncoder(dim_neck, dim_emb, freq)
        self.speaker_encoder = SpeakerEncoder(n_speakers, dim_emb)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = PostNet()

    def forward(self, x, src, trg):
        spk_emb_src = self.speaker_encoder(src)
        codes = self.content_encoder(x, spk_emb_src)  # [B, 2*32, N/freq]
        if trg is None:  # when calculating code semantic loss, return the codes only
            return torch.cat(codes, dim=-1)  # [B, 2*32, N]

        tmp = []
        for code in codes:  # [B, 2*32]
            # [B, 2*32] -> [B, 1, 2*32] -> [B, N/freq, 2*32]
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(2) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)  # expanded code [B, N, 2*32]

        # [B, 256] -> [B, 1, 256] -> [B, N, 256]
        spk_emb_trg = spk_emb_src.unsqueeze(1).expand(x.size(0), x.size(2), -1)
        decoder_inputs = torch.cat((code_exp, spk_emb_trg), dim=-1)

        # mel predicted by decoder
        mel_outputs = self.decoder(decoder_inputs)  # [B, N, 80]

        # residual connection
        mel_outputs_psnt = self.postnet(mel_outputs)  # [B, 80, N]
        mel_outputs_psnt = mel_outputs + mel_outputs_psnt.transpose(1, 2)  # [B, N, 80]

        mel_outputs = mel_outputs.unsqueeze(1)  # [B, 1, N, 80]
        mel_outputs_psnt = mel_outputs_psnt.unsqueeze(1)    # [B, 1, N, 80]

        # mel_outputs and mel_outputs_psnt are for mel reconstruction loss
        # codes
        return mel_outputs.transpose(2, 3), mel_outputs_psnt.transpose(2, 3), torch.cat(codes, dim=-1)  # [B, 2*N]


def build_model(config):
    generator = Generator(config.model.dim_neck,
                          config.model.dim_emb,
                          config.model.dim_pre,
                          config.model.freq,
                          config.data.n_speakers)

    optimizer = torch.optim.Adam([{'params': generator.parameters(), 'initial_lr': config.train.learning_rate}],
                                 lr=config.train.learning_rate,
                                 betas=config.train.betas,
                                 eps=config.train.eps)

    return generator, optimizer


if __name__ == '__main__':
    dim_neck = 32
    dim_emb = 256
    dim_pre = 512
    freq = 32
    n_speakers = 4

    generator = Generator(dim_neck, dim_emb, dim_pre, freq, n_speakers)
    x = torch.randn(2, 80, 1024)
    sid = torch.LongTensor([1])

    mel_outputs, mel_outputs_post, codes = generator(x, sid, sid)
    print(mel_outputs, mel_outputs_post, codes)
