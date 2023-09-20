# Kaiyu Li
# https://github.com/likyoo
#

import torch.nn as nn
import torch

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, sync_bn):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        if sync_bn:
            self.bn1 = nn.SyncBatchNorm(mid_ch)
        else:
            self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        if sync_bn:
            self.bn2 = nn.SyncBatchNorm(out_ch)
        else:
            self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels=21, hidden_channels=1, kernel_size=3):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        # forget gate
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # input gate
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # output gate
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # initialize model
        self.init_hidden()

    def forward(self, x, h, c):
        """
        :param x: (1,21,h,w)
        :param h: (1,1,h,w)
        :param c: (1,1,h,w)
        :return: c, h
        """
        # initialize if c,h is none
        if h is None:
            h = torch.zeros(1, self.hidden_channels, x.shape[2], x.shape[3]).cuda()
        if c is None:
            c = torch.zeros(1, self.hidden_channels, x.shape[2], x.shape[3]).cuda()

        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))

        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        # print('initialized successfully! default normal')
        # print('-' * 30)
        
        
class ClsHead(nn.Module):
    """
    Design a classification head to separate predictions to every stage.
    Because every stage has some pros and cons, and simple fusion layer will incline to most prediction conditions.
    input: 5 x H x W, cat(dsn1, dsn2, dsn3, dsn4, dsn5), after upsampled by deconv.
    return:
        selection: 5 x H x W, every channel only have some pixels activated as 1, the others are 0. We use this result to
        max map: supervise this output, use gt>0 may be better.
    """
    def __init__(self, in_channels, kernel_size=3, maxmode='max'):
        super(ClsHead, self).__init__()
        self.in_channels = in_channels
        self.cls_num = in_channels
        self.reduced_channels = 12
        self.maxmode = maxmode
        # conv layers
        self.conv_refine = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_bn = nn.BatchNorm2d(self.reduced_channels)
        self.conv_1x1 = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_1x1_bn = nn.BatchNorm2d(self.in_channels)
        if self.maxmode == 'max':
            self.maximum = torch.max
        elif self.maxmode == 'softmax':
            self.maximum = torch.softmax
        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_refine(x)
        x = self.relu(x)
        x = self.conv_bn(x)
        x = self.conv_1x1(x)
        x = self.conv_1x1_bn(x)
        if self.maxmode == 'max':
            x_out, indices = self.maximum(x, axis=1)
            selection = self._indices_to_selection(indices)
            # print(selection)
            x_out = torch.sigmoid(x_out)*selection
            return x_out
        elif self.maxmode == 'softmax':
            elwiseweight = self.maximum(x, dim=1)
            return elwiseweight

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _indices_to_selection(self, indices):
        selection = []
        for i in range(self.cls_num):
            selection.append((indices == i).float())
        selection = torch.stack(selection, dim=1)
        return selection

class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2, sync_bn=False):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 48     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0], sync_bn)
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1], sync_bn)
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2], sync_bn)
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3], sync_bn)
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4], sync_bn)
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0], sync_bn)
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1], sync_bn)
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2], sync_bn)
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3], sync_bn)
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0], sync_bn)
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1], sync_bn)
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2], sync_bn)
        self.Up2_2 = up(filters[2])

        self.Up2_1_1 = up(filters[2])
        self.Up2_2_1 = up(filters[2])
        
        self.Up3_1_1 = nn.ConvTranspose2d(filters[3], filters[3], 4, stride=4)

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0], sync_bn)
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1], sync_bn)
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0], sync_bn)

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.sa = SpatialAttention(kernel_size=3)
        self.ca1 = ChannelAttention(filters[1]*3, ratio=16)
        self.sa1 = SpatialAttention(kernel_size=3)
        self.ca2 = ChannelAttention(filters[2]*2, ratio=16)
        self.sa2 = SpatialAttention(kernel_size=3)
        self.ca3 = ChannelAttention(filters[3]*1, ratio=16)
        self.sa3 = SpatialAttention(kernel_size=3)
        # self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
        self.ca0_1 = ChannelAttention(filters[0], ratio=16 // 4)
        self.ca0_2 = ChannelAttention(filters[0], ratio=16 // 4)
        self.ca0_3 = ChannelAttention(filters[0], ratio=16 // 4)
        self.ca0_4 = ChannelAttention(filters[0], ratio=16 // 4)
        self.sa0_1 = SpatialAttention(kernel_size=3)
        self.sa0_2 = SpatialAttention(kernel_size=3)
        self.sa0_3 = SpatialAttention(kernel_size=3)
        self.sa0_4 = SpatialAttention(kernel_size=3)

        self.ca1_1 = ChannelAttention(filters[1], ratio=16//4)
        self.ca1_2 = ChannelAttention(filters[1], ratio=16//4)
        self.ca1_3 = ChannelAttention(filters[1], ratio=16//4)
        self.sa1_1 = SpatialAttention(kernel_size=3)
        self.sa1_2 = SpatialAttention(kernel_size=3)
        self.sa1_3 = SpatialAttention(kernel_size=3)

        self.ca2_1 = ChannelAttention(filters[2],ratio=16//4)
        self.ca2_2 = ChannelAttention(filters[2],ratio=16//4)
        self.sa2_1 = SpatialAttention(kernel_size=3)
        self.sa2_2 = SpatialAttention(kernel_size=3)
        
        #self.ca3_1 = ChannelAttention(filters[3],ratio=16//4)
        #self.sa3_1 = SpatialAttention(kernel_size=3)
        # self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        # self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        self.lstmcell_1 = ConvLSTMCell(input_channels=filters[0] * 4, hidden_channels=2, kernel_size=3)
        self.lstmcell_2 = ConvLSTMCell(input_channels=filters[1] * 3, hidden_channels=2, kernel_size=3)
        self.lstmcell_3 = ConvLSTMCell(input_channels=filters[2] * 2, hidden_channels=2, kernel_size=3)

        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)
        self.conv_final1 = nn.Conv2d(filters[1] * 3, out_ch, kernel_size=1)
        self.conv_final2 = nn.Conv2d(filters[2]*2, out_ch, kernel_size=1)
        self.conv_final3 = nn.Conv2d(filters[3]*1, out_ch, kernel_size=1)
        
        # cls head
        self.cls_head = ClsHead(8, maxmode='softmax')
        self.new_score_weighting = nn.Conv2d(10, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        x0_1 = self.ca0_1(x0_1) * x0_1
        x0_1 = self.sa0_1(x0_1) * x0_1
        x0_2 = self.ca0_2(x0_2) * x0_2
        x0_2 = self.sa0_2(x0_2) * x0_2
        x0_3 = self.ca0_3(x0_3) * x0_3
        x0_3 = self.sa0_3(x0_3) * x0_3
        x0_4 = self.ca0_4(x0_4) * x0_4
        x0_4 = self.sa0_4(x0_4) * x0_4

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        # the first scale

        x1_1 = self.Up1_1(x1_1)
        x1_2 = self.Up1_2(x1_2)
        x1_3 = self.Up1_3(x1_3)

        x1_1 = self.ca1_1(x1_1) * x1_1
        x1_1 = self.sa1_1(x1_1) * x1_1
        x1_2 = self.ca1_2(x1_2) * x1_2
        x1_2 = self.sa1_2(x1_2) * x1_2
        x1_3 = self.ca1_3(x1_3) * x1_3
        x1_3 = self.sa1_3(x1_3) * x1_3

        out1 = torch.cat([x1_1, x1_2, x1_3], 1)
        # the second scale

        x2_1 = self.Up2_1(x2_1)
        x2_1 = self.Up2_1_1(x2_1)
        x2_2 = self.Up2_2(x2_2)
        x2_2 = self.Up2_2_1(x2_2)

        x2_1 = self.ca2_1(x2_1) * x2_1
        x2_1 = self.sa2_1(x2_1) * x2_1
        x2_2 = self.ca2_2(x2_2) * x2_2
        x2_2 = self.sa2_2(x2_2) * x2_2
        out2 = torch.cat([x2_1, x2_2], 1)
        # the third scale
        
        x3_1 = self.Up3_1(x3_1)
        x3_1 = self.Up3_1_1(x3_1)
        out3 = x3_1
        # the fourth scale

        # intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        # ca1 = self.ca1(intra)
        # out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.ca(out) * out
        out = self.sa(out) * out
        #out = self.conv_final(out)
        score_0 = self.conv_final(out)

        out1 = self.ca1(out1)*out1
        out1 = self.sa1(out1)*out1
        #out1 = self.conv_final1(out1)
        score_1 = self.conv_final1(out1)

        out2 = self.ca2(out2)*out2
        out2 = self.sa2(out2)*out2
        #out2 = self.conv_final2(out2)
        score_2 = self.conv_final2(out2)
        
        out3 = self.ca3(out3)*out3
        out3 = self.sa3(out3)*out3
        out3 = self.conv_final3(out3)
        score_3 = out3

        hs_3 = None
        dsn_3 = out3
        hs_2, dsn_2 = self.lstmcell_3(out2, hs_3, dsn_3)
        hs_1, dsn_1 = self.lstmcell_2(out1, hs_2, dsn_2)
        hs_0, dsn_0 = self.lstmcell_1(out, hs_1, dsn_1)
        
        concat = torch.cat((dsn_0, dsn_1, dsn_2, dsn_3), 1)
        concat_score = torch.cat([score_0, score_1, score_2, score_3], 1)
        score_final = self.cls_head(concat_score)
        dsn_e = torch.sum(
            concat.view(-1, 4, 2, concat.size(-2), concat.size(-1)) * score_final.view(-1, 4, 2, score_final.size(-2),
                                                                                       score_final.size(-1)), axis=1)
                                                                                     
        concat = torch.cat((concat, dsn_e), 1)
        dsn_f = self.new_score_weighting(concat)
        
        return [(dsn_0, ), (dsn_1, ), (dsn_2, ), (dsn_3, ), (dsn_e, ), (dsn_f, )]
#        return (dsn_1, )

class Siam_NestedUNet_Conc(nn.Module):
    # SNUNet-CD without Attention
    def __init__(self, in_ch=3, out_ch=2, sync_bn=False):
        super(Siam_NestedUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0], sync_bn)
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1], sync_bn)
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2], sync_bn)
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3], sync_bn)
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4], sync_bn)
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0], sync_bn)
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1], sync_bn)
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2], sync_bn)
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3], sync_bn)
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0], sync_bn)
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1], sync_bn)
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2], sync_bn)
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0], sync_bn)
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1], sync_bn)
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0], sync_bn)

        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))
        return (output1, output2, output3, output4, output)
