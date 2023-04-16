import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dropout=0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=pad)
        self.dropout = dropout
        self.dropout2d = nn.Dropout2d(p=self.dropout)

    def initialize(self, inputs):
        device = inputs.device
        batch_size, _, height, width = inputs.size()

        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        self.cell_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, inputs, first_step=False):
        if first_step:
            self.initialize(inputs)

        combined = torch.cat([inputs, self.hidden_state], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_state = f * self.cell_state + self.dropout2d(i * g)
#         self.cell_state = f * self.cell_state + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        return self.hidden_state


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, device, dropout=0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = len(hidden_dim)
        self.device = device
        self.dropout = dropout

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(ConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=kernel_size, dropout=self.dropout))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], output_dim, kernel_size=1)

    def forward(self, input_x, input_frames=7, future_frames=5, output_frames=11,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The self-attention ConvLSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, C, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if the teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling 
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        assert len(input_x.shape) == 5
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames:
                input_ = input_x[:, t].to(self.device)
            elif not teacher_forcing:
                input_ = outputs[t-1]
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(self.device)
                input_ = input_x[:, t].to(self.device) * mask + outputs[t-1] * (1 - mask)
            first_step = (t == 0)
            input_ = input_.float()

            first_step = (t == 0)
            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)

        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1)  # (N, 37, H, W)
        return outputs
