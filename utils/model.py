import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): 
        lin = self.linear(x.permute(0, 2, 3, 1)) 
        lin = lin.permute(0, 3, 1, 2) 
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): 
        lin = self.linear(x.permute(0, 2, 3, 1)) 
        lin = lin.permute(0, 3, 1, 2) 
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)
        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == 'both':
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x): 
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)  
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)  
        elif self.pool_dim == 'both':
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)  

        if not self.pool_dim == 'both':
            x = self.conv1d1(x)               
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)               
        else:
            x = self.fc1(x)               
            x = self.relu(x)
            x = self.fc2(x)               

        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, pool_dim='freq'):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim
        self.n_basis_kernels = n_basis_kernels
        
        self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels,
                                     temperature, pool_dim)

        self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size),
                                   requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x): # x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ['freq', 'chan']:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)
        elif self.pool_dim == 'time':
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)
        elif self.pool_dim == 'both':
            softmax_attention = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    

        batch_size = x.size(0)
        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size) 

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
            
        output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))

        if self.pool_dim in ['freq', 'chan']:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == 'time':
            assert softmax_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * softmax_attention, dim=1)  

        return output


class SpectralGatingUnit(nn.Module):
    """Apply frequency weights derived from time-averaged features."""
    def __init__(self, channels, freq_bins, reduction=4):
        super().__init__()
        
        kernel_size = min(31, freq_bins)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_size = max(3, kernel_size)
        
        padding = (kernel_size - 1) // 2
        mid_channels = max(channels // reduction, 4)
        
        self.freq_model = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            
            nn.Conv1d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            
            nn.Conv1d(mid_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid() 
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, F = x.shape
        
        if T > 1:
            x_spectral_fingerprint = x.mean(dim=2) # [B, C, F]
        else:
            x_spectral_fingerprint = x.squeeze(2)

        spectral_weight = self.freq_model(x_spectral_fingerprint)
        
        spectral_weight_expanded = spectral_weight.unsqueeze(2)
        
        x_global = x * spectral_weight_expanded
        
        cat_feat = torch.cat([x, x_global], dim=1)
        
        alpha = self.fusion_gate(cat_feat) 
        
        return (1 - alpha) * x + alpha * x_global


class CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 temperature=31,
                 pool_dim='freq',
                 n_mels=128):
        super(CNN, self).__init__()
        self.n_filt = n_filt
        cnn = nn.Sequential()

        current_freq = n_mels 

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            
            if DY_layers[i] == 1:
                cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
                                                                   n_basis_kernels=n_basis_kernels,
                                                                   temperature=temperature, pool_dim=pool_dim))
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLu())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

            # 仅在频率维度足够大时使用，避免深层核大小问题
            if current_freq >= 4:
                cnn.add_module("tdsg{0}".format(i), 
                               SpectralGatingUnit(out_dim, freq_bins=current_freq))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            
            pool_k = pooling[i]
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pool_k))
            
            if isinstance(pool_k, (list, tuple)):
                f_stride = pool_k[1]
            else:
                f_stride = pool_k
            
            current_freq = current_freq // f_stride

        self.cnn = cnn

    def forward(self, x):    # x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x


class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 n_mels=128,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, 
                       n_mels=n_mels, **convkwargs)
        
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # input size : [bs, freqs, frames]
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1) # [bs, chan, frames, freqs]
        
        x = self.cnn(x)
        
        bs, ch, frame, freq = x.size()
        if freq != 1:
            # Flatten frequency into channel dimension
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frame, ch*freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # [bs, frames, chan]

        x = self.rnn(x) # [bs, frames, 2 * chan]
        x = self.dropout(x)

        strong = self.dense(x) # [bs, frames, n_class]
        strong = self.sigmoid(strong)
        
        if self.attention:
            sof = self.dense_softmax(x) 
            sof = self.softmax(sof) 
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1) # [bs, n_class]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak