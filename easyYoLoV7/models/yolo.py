import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

from models.heads import *

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None




class Model(nn.Module): # just yolov7 now
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False

        #self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.model = nn.Sequential( Conv(c_in=3, c_out=32, kernel_size=3, stride=1),    #0
                                    Conv(32, 64, 3, 2),   #1
                                    Conv(64, 64, 3, 1),   #2
                                    Conv(64, 128, 3, 2),  #3
                                    Conv(128, 64, 1, 1),  #4
                                    Conv(128, 64, 1, 1),  #5 from 3
                                    Conv(64, 64, 3, 1),   #6
                                    Conv(64, 64, 3, 1),   #7
                                    Conv(64, 64, 3, 1),   #8
                                    Conv(64, 64, 3, 1),   #9
                                    Concat(1),            #10 from [9, 7, 5, 4]
                                    Conv(256, 256, 1, 1), #11
                                    MP(),                 #12
                                    Conv(256, 128, 1, 1), #13
                                    Conv(256, 128, 1, 1), #14 from 11
                                    Conv(128, 128, 3, 2), #15
                                    Concat(1),            #16 from [15, 13]
                                    Conv(256, 128, 1, 1), #17
                                    Conv(256, 128, 1, 1), #18 from 16
                                    Conv(128, 128, 3, 1), #19
                                    Conv(128, 128, 3, 1), #20
                                    Conv(128, 128, 3, 1), #21
                                    Conv(128, 128, 3, 1), #22
                                    Concat(1),            #23 from [22, 20, 18, 17]
                                    Conv(512, 512, 1, 1), #24
                                    MP(),                 #25
                                    Conv(512, 256, 1, 1), #26
                                    Conv(512, 256, 1, 1), #27 from 24
                                    Conv(256, 256, 3, 2), #28
                                    Concat(1),            #29 from [28, 26]
                                    Conv(512, 256, 1, 1), #30   
                                    Conv(512, 256, 1, 1), #31 from 29
                                    Conv(256, 256, 3, 1), #32
                                    Conv(256, 256, 3, 1), #33
                                    Conv(256, 256, 3, 1), #34
                                    Conv(256, 256, 3, 1), #35
                                    Concat(1),              #36 from [35, 33, 31, 30]
                                    Conv(1024, 1024, 1, 1), #37
                                    MP(),                   #38
                                    Conv(1024, 512, 1, 1),  #39
                                    Conv(1024, 512, 1, 1),  #40 from 37
                                    Conv(512, 512, 3, 2),   #41
                                    Concat(1),              #42 from [41, 39]
                                    Conv(1024, 256, 1, 1),  #43
                                    Conv(1024, 256, 1, 1),  #44 from 42
                                    Conv(256, 256, 3, 1),   #45
                                    Conv(256, 256, 3, 1),   #46
                                    Conv(256, 256, 3, 1),   #47
                                    Conv(256, 256, 3, 1),   #48
                                    Concat(1),              #49 from [48, 46, 44, 43]
                                    Conv(1024, 1024, 1, 1), #50
                                    SPPCSPC(1024, 512, 1),  #51
                                    Conv(512, 256, 1, 1),   #52
                                    torch.nn.modules.upsampling.Upsample(None, 2, 'nearest'), #53
                                    Conv(1024, 256, 1, 1),  #54 from 37
                                    Concat(1),              #55 from [54, 53]
                                    Conv(512, 256, 1, 1),   #56
                                    Conv(512, 256, 1, 1),   #57 from 55
                                    Conv(256, 128, 3, 1),   #58
                                    Conv(128, 128, 3, 1),   #59
                                    Conv(128, 128, 3, 1),   #60
                                    Conv(128, 128, 3, 1),   #61
                                    Concat(1),              #62 from [61, 60, 59, 58, 57, 56]
                                    Conv(1024, 256, 1, 1),  #63
                                    Conv(256, 128, 1, 1),   #64
                                    torch.nn.modules.upsampling.Upsample(None, 2, 'nearest'), #65
                                    Conv(512, 128, 1, 1),   #66 from 24
                                    Concat(1),              #67 from [66, 65]
                                    Conv(256, 128, 1, 1),   #68
                                    Conv(256, 128, 1, 1),   #69 from 67
                                    Conv(128, 64, 3, 1),    #70
                                    Conv(64, 64, 3, 1),     #71
                                    Conv(64, 64, 3, 1),     #72
                                    Conv(64, 64, 3, 1),     #73
                                    Concat(1),              #74 from [73, 72, 71, 70, 69, 68]
                                    Conv(512, 128, 1, 1),   #75
                                    MP(),                   #76
                                    Conv(128, 128, 1, 1),   #77
                                    Conv(128, 128, 1, 1),   #78 from 75
                                    Conv(128, 128, 3, 2),   #79
                                    Concat(1),              #80 from [79, 77, 63] 
                                    Conv(512, 256, 1, 1),   #81
                                    Conv(512, 256, 1, 1),   #82 from 80
                                    Conv(256, 128, 3, 1),   #83
                                    Conv(128, 128, 3, 1),   #84
                                    Conv(128, 128, 3, 1),   #85
                                    Conv(128, 128, 3, 1),   #86
                                    Concat(1),              #87 from [86, 85, 84, 83, 82, 81]
                                    Conv(1024, 256, 1, 1),  #88
                                    MP(),                   #89
                                    Conv(256, 256, 1, 1),   #90
                                    Conv(256, 256, 1, 1),   #91 from 88
                                    Conv(256, 256, 3, 2),   #92
                                    Concat(1),              #93 from [92, 90, 51]
                                    Conv(1024, 512, 1, 1),  #94
                                    Conv(1024, 512, 1, 1),  #95 from 93
                                    Conv(512, 256, 3, 1),   #96
                                    Conv(256, 256, 3, 1),   #97
                                    Conv(256, 256, 3, 1),   #98
                                    Conv(256, 256, 3, 1),   #99
                                    Concat(1),              #100 from [99, 98, 97, 96, 95, 94]
                                    Conv(2048, 512, 1, 1),  #101
                                    RepConv(128, 256, 3, 1),    #102 from 75
                                    RepConv(256, 512, 3, 1),    #103 from 88
                                    RepConv(512, 1024, 3, 1),   #104 from 101
                                    Detect(nc=80, 
                                           anchors=[[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]],
                                           ch=[256, 512, 1024])                    #105 from [102, 103, 104]
                                  )
        self.froms = [-1, -1, -1, -1, -1,             #4
                      [3], -1, -1, -1, -1,              #9
                      [9, 7, 5, 4], -1, -1, -1, [11],   #14
                      -1, [15, 13], -1, [16], -1,       #19
                      -1, -1, -1, [22, 20, 18, 17], -1, #24
                      -1, -1, [24], -1, [28, 26],         #29
                      -1, [29], -1, -1, -1,               #34
                      -1, [35, 33, 31, 30], -1, -1, -1, #39
                      [37], -1, [41, 39], -1, [42],         #44
                      -1, -1, -1, -1, [48, 46, 44, 43], #49
                      -1, -1, -1, -1, [37],               #54
                      [54, 53], -1, [55], -1, -1,         #59
                      -1, -1, [61, 60, 59, 58, 57, 56], -1, -1, #64
                      -1, [24], [66, 65], -1, [67],     #69
                      -1, -1, -1, -1, [73, 72, 71, 70, 69, 68], #74
                      -1, -1, -1, [75], -1,             #79
                      [79, 77, 63], -1, [80], -1, -1,   #84
                      -1, -1, [86, 85, 84, 83, 82, 81],-1, -1,  #89
                      -1, [88], -1, [92, 90, 51], -1,   #94
                      [93], -1, -1, -1, -1,             #99
                      [99, 98, 97, 96, 95, 94], -1, [75], [88], [101], #104
                      [102, 103, 104]
                     ]
        print("froms:" , self.froms)

        self.save = []
        for i in self.froms:
            if i != -1:
                self.save += i
        print("self.save: ", self.save)




        self.names = [str(i) for i in range(80)]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        
        s = 256  # 2x min stride
        m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        check_anchor_order(m)
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases()  # only run once
        # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for i, m in enumerate(self.model):
            if self.froms[i] != -1:
                x = [x if j == -1 else y[j] for j in self.froms[i]]  # from earlier layers
                if len(x) == 1:
                    x = x[0]

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m, IKeypoint):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)

                np = sum([x.numel() for x in m.parameters()])
                mtype = str(m)[8:-2].replace('__main__.', '')  # module type
                print('%10.1f%10.0f%10.1fms %-40s' % (o, np, dt[-1], mtype))

            x = m(x)  # run
            
            y.append(x if i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0,1,2,bc+3)].data
            obj_idx = 2*bc+4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx+1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            b[:, (0,1,2,bc+3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                #print(f" switch_to_deploy")
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC, 
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv, 
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,  
                 Res, ResCSPA, ResCSPB, ResCSPC, 
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC, 
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC, 
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC, 
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC, 
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC, 
                     ResCSPA, ResCSPB, ResCSPC, 
                     RepResCSPA, RepResCSPB, RepResCSPC, 
                     ResXCSPA, ResXCSPB, ResXCSPC, 
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else: #m is MP
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
