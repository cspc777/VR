# reference https://github.com/pengr97/ConvLSTM

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, padding="SAME", stride=1, dilation=1, bias=True):
        """
        :param input_size: (h,w)
        :param input_dim: the channel of input xt
        :param hidden_dim: the channel of state h and c
        :param padding: add "SAME" pattern
        """

        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        # Because Pytorch has no "SAME" pattern, I add this pattern
        padding = padding if isinstance(padding, tuple) else (padding, padding)
        if padding[0] == "SAME":
            padding_h = ((self.input_size[0] - 1) * self.stride[0] - self.input_size[0] + self.kernel_size[0] + (self.kernel_size[0] - 1) \
                        * (self.dilation[0] - 1))//2
        if padding[1] == "SAME":
            padding_w = ((self.input_size[1] - 1) * self.stride[1] - self.input_size[1] + self.kernel_size[1] + (self.kernel_size[1] - 1) \
                        * (self.dilation[1] - 1))//2
        self.padding = (padding_h, padding_w)

        # in_channels, out_channels, kernel_size, stride=1, padding=0,
        # dilation=1, groups=1, bias=True, padding_mode='zeros'
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      stride=stride,
                      dilation=dilation,
                      bias=bias),
            # nn.ReLU()
        )

    def forward(self, xt, state):
        """
        :param xt: (b,c,h,w)
        :param state: include c(t-1) and h(t-1)
        :return: c_next, h_next
        """
        c, h = state

        # concatenate h and xt along channel axis
        com_input = torch.cat([xt, h], dim=1)
        com_outputs = self.conv(com_input)
        temp_i, temp_f, temp_o, temp_g = torch.split(com_outputs, self.hidden_dim, dim=1)

        i = torch.sigmoid(temp_i)
        f = torch.sigmoid(temp_f)
        o = torch.sigmoid(temp_o)
        g = torch.tanh(temp_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return c_next, h_next

    def init_state(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])).cuda())
    
class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers=1, padding="SAME", stride=1, dilation=1, bias=True, batch_first=True):
        """
        :param input_size: (h,w)
        :param input_dim: the channel of input xt in the first layer
        :param hidden_dim: the channel of state for all layers, this is a single number,
        we will extend it for the multi layers by transform it to a list
        :param num_layers: the layer of ConvLSTM
        :param padding: the padding of all th layers
        :param batch_first: batch is the first dim if true
        """

        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = [hidden_dim]*self.num_layers if not isinstance(hidden_dim, list) else hidden_dim
        self.kernel_size = [kernel_size]*self.num_layers if not isinstance(kernel_size, list) else hidden_dim
        self.padding = [padding]*self.num_layers if not isinstance(padding, list) else padding
        self.batch_first = batch_first

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=input_size,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          padding=self.padding[i],
                                          stride=stride,
                                          dilation=dilation,
                                          bias=bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_x, state=None):
        """
        :param input_x: the total input data, (batch,time_steps,channel,height,weight)
        :param state: the initial state include hidden state h and cell state c
        :return: outputs (batch,time_steps,channel,height,weight), last_state (shape of h is the same with c, shape: (b,c,h,w))
        """

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_x = input_x.permute(1, 0, 2, 3, 4)

        batch_size = input_x.size(0)
        time_step = input_x.size(1)

        state = self.init_state(batch_size)

        for layer in range(self.num_layers):
            c, h = state[layer]
            t_output = []
            for t in range(time_step):
                cur_input = input_x[:, t, :, :, :]
                c, h = self.cell_list[layer](xt=cur_input, state=(c, h))
                t_output.append(h)

            layer_output = torch.stack(t_output, dim=1)
            input_x = layer_output

        # take the last layer's output and state as output
        outputs = layer_output
        last_state = (c, h)

        return outputs, last_state

    def init_state(self, batch_size):
        init_state = []
        for layer in range(self.num_layers):
            init_state.append(self.cell_list[layer].init_state(batch_size))
        return init_state