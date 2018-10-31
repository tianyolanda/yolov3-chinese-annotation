# 存放yolo的网络结构
from __future__ import division
from util import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img,(416,416))
    img_ = img[:,:,::-1].transpose((2,0,1)) # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0 # 添加一个channel在0的位置, 为了batch | 然后再归一化图像
    img_ = torch.from_numpy(img_).float() # 转化为float
    img_ = Variable(img_)
    return img_


class EmptyLayer(nn.Module):
    # 自定义的layer: 需要继承父类 nn.Module
    def __init__(self): # init函数是类的初始化方法,在创建类的实例的时候会调用此方法
        # 类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是 self
        # self 代表的是类的实例，代表当前对象的地址
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    # 前向传播函数两个作用: 1. 计算输出
    #                   2. 简化中间传输output的步骤,以便实现一些多尺度跨层叠加操作(需要知道上一层输出维度的层),等等
    def forward(self, x, CUDA):
        # CUDA如果是TRUE, 就可以使用GPU加速
        detections = []
        modules = self.blocks[1:]
        outputs = {} # 因为route层和shortcut层需要前一层的output map,所以我们缓存所有中间产生的feature map. 使用键值对的方式存储, 键是层数, 值是feature map

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)  # module_list[i]就是第i个layer,比如现在第i层是某一个convolution layer
                                            # 然后(x)的意思是将x这个实例,丢到这个网络层里去进行计算!
                                            # pytorch调用这些module的时候, 都需要把对哪一层做操作丢进去
            elif module_type == "route": # 借鉴Resnet融合细粒度特征的思想
                # 路由层根据layer不同有两种情况: 1. [-4]: 输出feature map
                                         #   2. [-1,61]: 使用torch.cat((第-1层,第61层),1)来 concatenate two feature maps

                layers = module["layers"]
                layers = [int(a) for a in layers]

                if(layers[0]) > 0: # 这一步基本不会做的..
                    layers[0] = layers[0] - i # 减完之后肯定是个负数了

                if len(layers) == 1: # 如果layers有1个值, 也就是情况1,直接输出该层的feature map
                    x = outputs[i + (layers[0])] # 倒数第几层转化为正数第几层

                else: # 如果layers有2个值,也就是情况2, 输出两层feature map按照深度方向concat
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i # 负数

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1,map2),1)

            elif module_type == "shortcut":
                from_ = int(module["from"]) # 这个from_是相对于当前层的层数,比如-3表示:当前层前面第3层
                x = outputs[i-1] + outputs[i + from_] # Resnet里面的跨层链接, 把两层的值相加, 第三层和第一层短接

            elif module_type == "yolo":
                anchor = self.module_list[i][0].anchors # [0]表示 DetectionLayer这个类, .anchor表示访问类的变量
                inp_dim = int(self.net_info["height"])

                num_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x, inp_dim, anchor, num_classes, CUDA)

                if not write:
                    # write flag用于指示我们是否遇到了第一次检测,如果write为0，则表示收集器尚未初始化。如果它是1，则表示collector初始化，我们可以继续concatenate检测映射。
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections,x),1) # 两张图并列放置

            outputs[i] = x

        return detections

    def load_weights(self,weightfile):
        # 权重文件是全为浮点数,按顺序依次排列,没有任何其他的指示字符, 我们要自己按照网络结构的顺序解析它,(一旦错了万劫不复哦,因为没人提醒你加载错了,比如把conv层的权重加载到bn层也不会报错,只会导致识别效果凉凉)
        # 解析权重,首先要明白权重的浮点数的排列和意义,yolo是全卷积网络,它的权重就是卷积核部分(以及 有/无batch normalization)
        # 卷积核分为两种: 1.带bn的卷积核   2.不带bn的卷积核
        # 1.带bn的卷积核组成(按顺序): bn_biases, bn_weights, bn_running_mean, bn_running_var, conv_weights
        # 2.不带bn的卷积核组成(按顺序): conv_biases, conv_weights

        fp = open(weightfile,"rb")

        # 前5个值是标头
        #  1. 主版本信息
        # 2.3.4.5...各种版本信息和image?
        #The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen

        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            # 只有convolution层加载权重
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"]) # 这里因为在convolutional模块里面包含batch_normalize=1这样的东西,这里的判断是在判断该conv模块里面有/没有bn层
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize) :
                    bn = model[1]

                    num_bn_biases = bn.bias.numel() # 找到bn层的weight数量, 32

                    #加载权重
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]) # 从第ptr个浮点数加载到第ptr+num_bn_bias个浮点数 : 加载bias
                    ptr += num_bn_biases # 指针移动(bn层的四个参数,每个都移动num_bn_bias这个长度,是因为他们的参数数量都一样,等于神经元个数)
                                        # ax+b
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]) # 加载weights
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]) # 均值
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]) #
                    ptr += num_bn_biases

                    # 投影loaded weights 到 model weithts的dim中
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy data到model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                # 如果本conv模块没有bn层,则只是简单的加载conv层的bias (conv的权重在最后一起加载)
                else:
                    # bias的数量
                    num_biases = conv.bias.numel()

                    #加载权重
                    conv_biases = torch.from_numpy(weights[ptr: ptr+num_biases])
                    ptr = ptr + num_biases

                    # reshape
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # 复制data
                    conv.bias.data.copy_(conv_biases)

                # 加载完bn层权重再开始加载conv层的权重
                num_weights = conv.weight.numel() # 计本层conv卷积核自己包含多少个权重
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights]) # 往后挪着,读取weight file里的浮点数
                ptr = ptr + num_weights # 指针后裔

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



def parse_cfg(cfgfile):
    #这部分是在解析cfg文件,理解里面的text并按模块存储到blocks这个list里面,list里有很多个字典{},每个字典存储了一个block,也就是一个小conv或者route模块
    file = open(cfgfile,'r')
    lines = file.read().split('\n') # 将文件中的所有行,存在一个list里面
    lines = [x for x in lines if len(x) > 0 ]  # 将非空行存在x里面
    lines = [x for x in lines if x[0] !='#'] # 去掉其中的注释行
    lines = [x.rstrip().lstrip() for x in lines] # # 去掉每行的头尾符
    block = {} # block是个字典,存储每个block的信息. 字典通过键值对访问: blcok["键"] = "值"
    blocks = [] # blocks是个列表list,存储所有的block. 列表通过下标访问, blocks[1],block[1:],etc

    for line in lines:
        if line[0] == "[": # 如果开始了一个新的block
            if len(block) != 0 : # 但是老的block还没有被存储和删去
                blocks.append(block) # 首先存储老的block到blocks中
                block = {} # 再把旧的内容删去
            block["type"] = line[1:-1].rstrip() #将[]内的字,记录到block"type"中
        else:
            key,value = line.split("=") # 将键值对拆分,比如 stride = 1 变成 'stride':'1' , 注意:如果是layer=1,2 这种,一个键对应两个值, 就会变成 'layer':'1,2'
            block[key.rstrip()] = value.lstrip() # key去头,作为标签; value去尾, 作为值

    blocks.append(block) # 最后要把最后一个block给加入到blocks里面去

    return blocks


#接下来需要依次解析这些block,并把他们生成pytorch module, 除了convlution和upsample已经被pytorch定义好了,其他的都需要自己实现
def create_modules(blocks):
    net_info = blocks[0] #输出的是第一个block--net的全部内容
    module_list = nn.ModuleList()
    prev_filters = 3 #记录上一层卷积核数量(特征图深度),以决定这一层卷积核深度. 初始化为3:初始RGB三通道
    output_filters = [] #路由层（route layer）从前面层得到特征图（可能是拼接的）。如果在路由层之后有一个卷积层，那么卷积核将被应用到前面层的特征图上，精确来说是路由层得到的特征图。因此，我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。

    for index, x in enumerate(blocks[1:]): # 从blocks的第二个开始,挨个block排查,index是enumerate自带的计数器,从0开始数,这里index表示是第几层. x是迭代的block的内容,字典
        module = nn.Sequential() # 因为每个模块内部也有很多层组成,所以用sequential串联他们. 最后把所有的module加入到module list中去
        if(x["type"] == "convolutional"):
            # 获得该层的信息,挨个把block里的东西存入下面几个变量中
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2 # 取整除:返回商的整数部分（向下取整. e.g. 9//2 输出结果 4 , 9.0//2.0 输出结果 4.0
            else:
                pad = 0

            # 添加这个convolution层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv) # 前面是name, 后面是child module

            # 添加batch normalize层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters) # filters是卷积层输出out_channels的数量,作为这里的输入
                module.add_module("batch_norm_{0}".format(index),bn)

            #检查激活函数的类型:在YOLO中只有 Linear 和 Leaky ReLU 两种
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(index),activn)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode= "bilinear")
            module.add_module("upsample_{0}".format(index),upsample)

#在路由层之后的卷积层会把它的卷积核应用到之前层的特征图（可能是拼接的）上。以下的代码更新了 filters 变量以保存路由层输出的卷积核数量。
        elif (x["type"] == "route"):
            # 路由层的layer属性值 一共有2种可能: 1. 一个值[-4]:输出(当前层往前数第四层)的feature map ,
            #                               2. 两个值[-1,61]: 将 当前层的前一层 和 第61层 的feature map 沿depth方向concatenated, 输出

            x["layers"] = x["layers"].split(',') # 也就是把 'layers': '-1, 61' 里面的-1 和 61 分开来, 这时候新的x["layer"] = [-1,61]

            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - index # index代表了目前层的层数,
            if end > 0:
                end = end - index

            #这里毫无动作
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),route)

            # 这里更新filters的数量以供route层接下来的conv层用: 在路由层之后的卷积层会把它的卷积核应用到之前层的特征图（可能是拼接的）上。以下的代码更新了 filters 变量以保存路由层输出的卷积核数量。
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end] # 最后这个filters就是prev_filters,同时也是 output_filters[index]
            else:
                filters = output_filters[index + start]

        # shortcut 对应的 skip connection
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),shortcut)

        elif (x["type"] == "yolo"):
            # mask表示: 要使用anchor中的第几对儿长宽 作为anchor
            mask = x["mask"].split(",") # mask=[3,4,5]
            mask = [int(x) for x in mask] # 对于mask中的每个数x, mask=它的整数

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)] # 步长为2的在两个两个走,因为anchor是两个两个为一组(表示长和宽),所以解析方法才是这样
            anchors = [anchors[i] for i in mask] # 取出mask中对应的anchor

            detection = DetectionLayer(anchors) #真正的detection操作层
            module.add_module("Detection_{}".format(index), detection)

        # 每个模块 识别完之后,加在总的modulelist中,同事更新一些参数
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info,module_list)



# model = Darknet("cfg/yolov3.cfg") # 类对象初始化
# model.load_weights("yolov3.weights")
# inp = get_test_input()
# pred = model(inp,torch.cuda.is_available())
# print(pred)

# m = parse_cfg('cfg/yolov3.cfg')
# print(create_modules(m))
