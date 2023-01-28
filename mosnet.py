import numpy as np
from torch import nn
import julius
import torch.nn.functional as F
import torch

class TimeDistributedTorch(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributedTorch, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output

class Conv2dPadded(nn.Module):
    # A modified version of nn.Conv2d such that output is consistent with output of keras Conv2D layer with padding "same" with stride>1 support
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dPadded, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.filter_height, self.filter_width = kernel_size
        self.stride_height, self.stride_width = stride
  
    def forward(self, x):
        _, _, in_height, in_width = x.shape
        if (in_height % self.stride_height == 0):
            pad_along_height = max(self.filter_height - self.stride_height, 0)
        else:
            pad_along_height = max(self.filter_height - (in_height % self.stride_height), 0)
        if (in_width % self.stride_width == 0):
            pad_along_width = max(self.filter_width - self.stride_width, 0)
        else:
            pad_along_width = max(self.filter_width - (in_width % self.stride_width), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        x = self.conv(x)
        return x

class MOSNet(nn.Module):
    def __init__(self, frame_length=1024, dropout=0, device='cuda'):
        super(MOSNet, self).__init__()
        self.crop_samples = frame_length - 4
        # STFT
        self.frame_length = frame_length
        self.hann_win = torch.hann_window(self.frame_length).to(device)
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), padding=(1, 1)), nn.ReLU(),
            Conv2dPadded(16, 16, (3, 3), (1, 3)), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1)), nn.ReLU(),
            Conv2dPadded(32, 32, (3, 3), (1, 3)), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), padding=(1, 1)), nn.ReLU(),
            Conv2dPadded(64, 64, (3, 3), (1, 3)), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), padding=(1, 1)), nn.ReLU(),
            Conv2dPadded(128, 128, (3, 3), (1, 3)), nn.ReLU())

        self.blstm1 = nn.LSTM(7 * 128, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # FC
        self.flatten = TimeDistributedTorch(nn.Flatten(), batch_first=True)
        self.dense1 = nn.Sequential(
            TimeDistributedTorch(nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU()), batch_first=True),
            nn.Dropout(dropout))

        # frame score
        self.frame_layer = TimeDistributedTorch(nn.Linear(128, 1), batch_first=True)
        # avg score
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def getFtrMaps(self, forward_input):
        # Returns a list with outputs of conv layers
        # Expected forward_input: waveform signal, shape: [batch, length]
        out_list = []
        input_stft = torch.stft(forward_input, self.frame_length, hop_length=self.frame_length//4, win_length=self.frame_length, window=self.hann_win)
        real = input_stft[..., 0]
        imag = input_stft[..., 1]
        input_stft = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).float()
        input_stft = input_stft[..., 2:-2].unsqueeze(1).permute((0, 1, 3, 2)) # remove 2 samples from start and end to be consistent with tf model
        conv1_output = self.conv1(input_stft)
        out_list.append(conv1_output)
        conv2_output = self.conv2(conv1_output)
        out_list.append(conv2_output)
        conv3_output = self.conv3(conv2_output)
        out_list.append(conv3_output)
        conv4_output = self.conv4(conv3_output)
        out_list.append(conv4_output)
        return out_list
    
    def forward(self, forward_input):
        # Expected forward_input: waveform signal, shape: [batch, length]
        input_stft = torch.stft(forward_input, self.frame_length, hop_length=self.frame_length//4, win_length=self.frame_length, window=self.hann_win)
        real = input_stft[..., 0]
        imag = input_stft[..., 1]
        input_stft = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).float()
        input_stft = input_stft[..., 2:-2].unsqueeze(1).permute((0, 1, 3, 2)) # remove 2 samples from start and end to be consistent with tf model
        conv1_output = self.conv1(input_stft) 
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        
        # reshape
        conv4_output = conv4_output.permute(0, 2, 3, 1)
        conv4_output = torch.reshape(conv4_output, (conv4_output.shape[0], -1, 7 * 128))

        # blstm
        blstm_output, (h_n, c_n) = self.blstm1(conv4_output)
        blstm_output = self.dropout(blstm_output)

        flatten_output = self.flatten(blstm_output)
        fc_output = self.dense1(flatten_output)
        frame_score = self.frame_layer(fc_output)

        avg_score = self.average_layer(frame_score.permute(0, 2, 1))
        return torch.reshape(avg_score, (avg_score.shape[0], -1)), frame_score


class MOSNetLoss(nn.Module):
    def __init__(self, model_weights, frame_length=1024, device='cuda'):
        super(MOSNetLoss, self).__init__()
        self.mosnet = MOSNet(frame_length=frame_length, device=device)
        self.mosnet.load_state_dict(torch.load(model_weights))
        self.mosnet.train().to(device)
    
    def ftrLoss(self, x, ref):
        # Compute L1 loss on feature maps:
        ftrs_x = self.mosnet.getFtrMaps(x)
        ftrs_ref = self.mosnet.getFtrMaps(ref)
        loss = 0
        for i in range(len(ftrs_x)):
            loss += F.l1_loss(ftrs_x[i], ftrs_ref[i]).mean()
        loss /= len(ftrs_x)
        return loss
    
    def forward(self, x, ref=None):
        # Expected input: waveform signal, shape: [batch, length]
        avg_mos_score, mos_score = self.mosnet(x)
        if not(ref is None):
            avg_mos_score_ref, mos_score_ref = self.mosnet(ref)
        return 5 - avg_mos_score.mean() if ref is None else ((mos_score_ref - mos_score) ** 2).mean()
