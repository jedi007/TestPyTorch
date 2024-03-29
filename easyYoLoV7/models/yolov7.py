import torch
from models.common import *
from models.heads import Detect

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush' ]

model = nn.Sequential(  Conv(c_in=3, c_out=32, kernel_size=3, stride=1),    #0
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
                        Detect(nc=len(names), 
                                anchors=[[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]],
                                ch=[256, 512, 1024])                    #105 from [102, 103, 104]
                        )

froms = [-1, -1, -1, -1, -1,             #4
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
print("froms:" , froms)



def create():
    save = []
    for i in froms:
        if i != -1:
            save += i
    save = sorted(set(save))
    print("save: ", save)

    return model, froms, save, names