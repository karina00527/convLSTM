# 具体实现
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels = self.in_channels + self.out_channels,
            out_channels = 4 * self.out_channels,
            kernel_size = self.kernel_size,
            padding = self.padding,
            bias = self.bias)

    def forward(self, x, h, c):
        # x: (B,C,H,W), h/c: (B,out_channels,H,W)
        combined = torch.cat([x, h], dim=1)  # (B,C+out_channels,H,W)
        combined_conv = self.conv(combined)
        i, f, g, o = torch.chunk(combined_conv, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMSeg(nn.Module):
    """用于像素级分类的 ConvLSTM"""
    def __init__(self, in_channels, out_channels, kernel_size, num_layers,
                 num_classes=3, batch_first=True, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.num_classes = num_classes

        # ConvLSTM layers
        cell_list = []
        cur_input = in_channels
        for i in range(num_layers):
            cell_list.append(ConvLSTMCell(
                in_channels=cur_input,
                out_channels=out_channels[i],
                kernel_size=kernel_size[i],
                bias=bias))
            cur_input = out_channels[i]
        self.cell_list = nn.ModuleList(cell_list)
        
        # 输出头：像素级分类
        self.out_conv = nn.Conv2d(out_channels[-1], num_classes, 1)
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.out_channels[i], *image_size,
                          device=self.out_conv.weight.device)
            c = torch.zeros(batch_size, self.out_channels[i], *image_size,
                          device=self.out_conv.weight.device)
            init_states.append((h, c))
        return init_states

    def forward(self, x):
        # x: (B,T,C,H,W) if batch_first else (T,B,C,H,W)
        if not self.batch_first:
            x = x.permute(1,0,2,3,4)  # (B,T,C,H,W)
        B, T = x.size(0), x.size(1)
        image_size = x.size()[3:]
        
        # 初始化隐状态
        hidden_states = self._init_hidden(B, image_size)
        
        # ConvLSTM编码
        layer_output = None
        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = self.cell_list[layer_idx](
                    x=cur_layer_input[:, t, ...],
                    h=h, c=c)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
        
        # 像素级分类（用最后一个时间步）
        last_h = layer_output[:, -1]  # (B,C,H,W)
        logits = self.out_conv(last_h)  # (B,num_classes,H,W)
        return logits
