# # yolov3 inference源码阅读--tianyolanda(2018.11)
#
# ## 以detect.py,也就是main函数开始
# 1. 用户输入,传递一堆参数
# 2. 构建yolo模型结构
# >  model = Darknet(args.cfgfile)
# 3. 加载权重(训练好的权重)
# >  model.load_weights(args.weightsfile)
# 4. 根据用户输入,修改模型参数(比如输入图像的长宽)
# 5. 读取图像,存储原图大小(未来还原用),预处理图像(resize等)
# 6. 将所有图片叠起来变成一个batch,一起检测
# 7. 检测1: 首先将batch送入model中,前向传播,经过许多卷积层,最后输出一个特征图(B x 10647 x 85维) [以608*608的图为例]
# > 每张图片都检测出10647个结果,B 是batch size(一共有多少张图片), 85 = 物体类别80 + (x,y,w,h) + 本bbox是否有物体的confidence
# 8. 检测2: 对每张图的10647个检测结果进行NMS,挑选出有效bbox框,存成一个tensor(实现该功能的函数叫write_results())
# > 这里注意,由于每幅图的NMS的confidence阈值不一样,因此要把每一张图从batch中拆开单独过一遍for循环.
# >  NMS原理--每幅图都关注:同类型里最高分的bbox,并以他为基准判断其他同类型的bbox是否与他重复,是否删掉)
# 9. 对每张图的检测,统计时间和bbox数目,输出
# 10. 画框
# > 之前resize的时候填充/拉抻过图片,这里首先裁掉画到原图像外部的bbox框
# > 将bbox框映射到原图,画上框,并存储检测后的新图
# 11. 统计运算时间等数据

from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def letterbox_image(img, inp_dim):
    # resize image and padding it
    # 自定义的resize函数,目的是resize(按照原图横纵比) + padding(剩下的像素用128填充),把图片变成inp_dim大小,再输入到YOLO
    img_w, img_h = img.shape[1], img.shape[0]
    w,h = inp_dim

    # 计算按照原图横纵比缩小至inp_dim以内,最大的图的size(是长=inp_dim 还是宽=inp_dim,得看这张图的长和宽哪个更长)
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))

    resized_image = cv2.resize(img,(new_w,new_h), interpolation=cv2.INTER_CUBIC) # 双三次插值

    canvas = np.full((inp_dim[1], inp_dim[0],3), 128) # 新建一个像素值全是128的3通道图

    canvas[(h-new_h)//2: (h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, : ] = resized_image #替换resize_image到这张图里(也就是空白处会被128padding)

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    :param img: 图像
    :param inp_dim: 输入维度
    :return: Variable
    """

    # img = cv2.resize(img,(inp_dim,inp_dim))
    img = (letterbox_image(img, (inp_dim, inp_dim)))

    img = img[:,:,::-1].transpose((2,0,1)).copy() # img[:,:,::-1].transpose((2,0,1)) 是把维度交换按照(2,0,1)的顺序交换
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0) # BGR -> RGB (OpenCV load的图像是BGR形式
    return img



def arg_parse():
    """
    传递参数
    其中 dest="" 是参数名称
    """
    parser = argparse.ArgumentParser(description='YOLO V3 Detection Module')
    parser.add_argument("--images",dest = 'images' ,help ="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest = 'det', help =
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "608", type = str)
    return parser.parse_args()

if __name__ ==  '__main__':

    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes("data/coco.names")

    print("Loading network.........")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    # Detection phase

    # Read the image from the disk, or the images from a directory. The paths of the image/images are stored in a list called imlist.
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)] # imlist是图片的路径集合
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath(','),images))
    except FileNotFoundError:
        print("No file or dictory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()
    loaded_ims = [cv2.imread(x) for x in imlist] # opencv读取图片,存入loaded_ims

    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
    # map()是Python内置的高阶函数,它接收一个函数f和一个list，并通过把函数 f 依次作用在 list 的每个元素上，返回一个list的遍历对象。如果想得到一个list列表，则用list(map())进行强制转换。
    # 本段的含义是: prep_image()这个函数,将后面的两项作为函数的输入. 函数的所有返回值存成list,赋值给im_batches

    # 存储原图的大小(以便检测万再将bbox映射回原图)  List containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist)
        im_batches = [torch.cat(im_batches[ i*batch_size : min((i+1) * batch_size,
                                len(im_batches))]) for i in range(num_batches)]
        # 把batch拆开,堆叠(cat)起来

    write = 0
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        # load the image
        # 每个batch
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        prediction = model(Variable(batch, volatile = True), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

        end = time.time()

        # 无检测情况:
        # 如果batch的write_results函数的输出是int(0)，意味着[没有检测]，我们使用continue跳过rest循环。
        if type(prediction) == int:

            for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size,len(imlist))]):
                im_id = i*batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds" . format(image.split("/")[-1], (end-start)/batch_size))
                print("{0:20s} {1:s}".format("Object Detected:", ""))
                print("---------------------------------------------------")
            continue

        prediction[:,0] += i*batch_size # #transform the atribute from index in batch to index in imlist
        #　prediction是每一个batch的结果输出(换一个batch就会重置一次)
        # 因此每个batch的prediction要把自己的batch标号加上
        # 随后该batch的prediction再存入 output里--所有batch的全部预测结果

        if not write:
            output = prediction
            write = 1

        else:
            output = torch.cat((output,prediction))

        # 有检测情况:
        # 以上已经获得了本batch的预测结果,存在了output里面
        # 下面进行 结果的解析 和 展示

        for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size+im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id] # x也就是本张im_id图,检测出的所有bbox信息序列, x[-1]存着classname
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end-start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("--------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()

    # 目前为止预测结果都预测好了,存在output中,
    # output[0]是batch标号,从[1:5]是bbox框的信息, [-1]是classname
    # 但预测的结果是基于inp_dim这个图像大小的,也就是是resize过后的, 我们需要把结果映射到原图上

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        # 裁掉那些,bbox在图片外的bbox
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0]) # 压缩限制函数: input, min,max
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()

    def write_draw(x, results):
        # 画框函数,把框画到图片上
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    # loaded_ims是原图集合
    list(map(lambda x: write_draw(x, loaded_ims), output))    # 定义了一个lambda函数,反正output就是x的实值输入write_draw()里面

    # 最后，将带有检测的图像写入det_names中的地址
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

    list(map(cv2.imwrite, det_names, loaded_ims))
    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")
    torch.cuda.empty_cache()























