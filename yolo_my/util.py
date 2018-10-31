# 存放辅助函数

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def load_classes(namesfile):
    fp = open(namesfile,"r")
    names = fp.read().split("\n")[:-1]
    return names

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):

    # 输入的prediction是[经过YOLO网络得到的预测到结果]
    # 本函数的目的是将prediction变换维度,变成一个方便查看结果和理解的tensor!!!!
    # 因此本函数全是在做维度转换

    # inp_dim 是输入图像的像素大小,e.g. 608*608, 那inp_dim = 608
    # num_classes = 80
    # anchors 是一个list,存储本次使用的3种anchor的像素值,e.g.[(116,90),(156,198),(373,326)]

    # 本函数最终输出的变换向量(也就是预测的bbox信息集合)的每一列是一种属性,一共 (5+c)列:
    # 5 : x,y,w,h,conf(当前bbox内存在物体的可能性)
    # c = 80 : 类别数目
    # 每行是一个预测的bbox,每三行表示一个格子(因为一个格子中预测3个obj)


    batch_size = prediction.size(0) # e.g. prediction的维度: [1,255,19,19]
    stride = inp_dim // prediction.size(2) # stride是步长 = 608 / 19 = 32
    gride_size = inp_dim // stride # 每个格子的大小(真实像素大小) = 19, 其实就是prediction.size(2)
    bbox_attrs = 5 + num_classes # 5 + 80 = 85
    num_anchors = len(anchors) # 几个anchor: 3


    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, gride_size*gride_size) # 拉抻向量--[1,255,361], 把原来的19*19合并为361, 去掉一个维度
    prediction = prediction.transpose(1,2).contiguous() # 变换向量--[1,361,255], 把255和361的位置调个儿
    prediction = prediction.view(batch_size, gride_size*gride_size*num_anchors, bbox_attrs) # 拉抻向量--[1,1083,85] : 361*3=1083, 255/3=85 ; 满足了(bbox数目:3*19*19)个行, 5+c个列
    # 目前为止,prediction[1,1083,85]的第一维表示batchsize,第二维表示哪个bbox,第三维表示该bbox属性

    anchors = [ (a[0] / stride, a[1] / stride) for a in anchors] # 把 anchor的原始长宽像素分别除以stride, 目的是变换到[以当前特征图大小为基准的](相对于19*19的)

    # 将预测坐标限定在(0,1)之间 sigmoid centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) # 将bbox的属性--中心的x坐标,经过simoid函数限制在(0,1)之间
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) # 将bbox的属性--中心的y坐标,经过simoid函数限制在(0,1)之间
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) # 将bbox的属性--存在obj的可能性:conf,经过simoid函数限制在(0,1)之间

    # 预测坐标微调
    grid = np.arange(gride_size) # arange(19) = [0,1,2,...,18]
    a,b = np.meshgrid(grid,grid) # a:        [[0,1,2,...,18]
                                #            [0,1,2,...,18]
                                #            .............
                                #            [0,1,2,...,18]]
                                #  注:一共是19*19的二维矩阵
    # b:        [[0,0,0,...,0]
    #            [1,1,1,...,1]
    #            [2,2,2,...,2]
    #            .............
    #            [18,18,18,...,18]]
    #  注:一共是19*19的二维矩阵

    x_offset = torch.FloatTensor(a).view(-1,1) # 是一个[361,1]维向量,361行,1列,数据值是(竖着): 0,1,2,..18,0,1,2,...,18,...,18 (一共361个)
    y_offset = torch.FloatTensor(b).view(-1,1) # [361,1]: 0,0,0,....,1,1,1,.....,18

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        prediction = prediction.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    # torch.cat((x_offset,y_offset),1)按列方向堆叠x_offset和y_offset,输出维度: [361*2, 2]
    # repeat(1,num_anchors)沿列方向复制3次,输出维度: [361*2, 6]
    # view(-1,2) 将向量维度变换成[361*2*3, 2]
    # unsqueeze(0) 添加一维向量 [1, 361*2*3, 2] = [1, 1083, 2]

    prediction[:,:,:2] += x_y_offset
    # 将prediction[:,:,0和1]是网络预测出的bbox框中心坐标x和y(以本格子的左上角为原点),需要微调后输出:加上x_y_offset(网格偏移)
    # 目的是,之前x和y已经被限制在本网格内部(0,1)之间了,现在要加上它所在的是第几个网格这个偏移量,也就是将其换算成19*19这个map里的绝对坐标

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(gride_size*gride_size,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors # prediction[:,:,2和3]是网络预测出的bbox框的w,h,需要微调后输出: 先求指数幂,再乘以anchors

    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes])) #每个classes的score也限制在0,1之间,使用sigmoid是因为某些类不是互斥的(女人/人),因此不选择softmax

    prediction[:,:,:4] *= stride # prediction[:,:,1和2和3和4]将bbox的x,y,w,h分别乘上stride, 从 19*19 恢复到原始图的inputsize 608*608

    return prediction


def bbox_iou(box1,box2):
    """

    :param box1:
    :param box2:
    :return:the IOU of two bboxes
    """
    # 获取bboxes的坐标
    b1_x1,b1_y1,b1_x2,b1_y2 = box1[:,0],box1[:,1],box1[:,2],box1[:,3]
    b2_x1,b2_y1,b2_x2,b2_y2 = box2[:,0],box2[:,1],box2[:,2],box2[:,3]

    # 获取相交矩形的坐标 get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.max(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    # Intersection area 相交区域
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union area : b1,b2各自的区域相加 - 相交区域 = 重叠区域
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # 这个函数是从 B x 10647 x 85 维的prediction里面,通过NMS筛选出真正的object!!,从10647中去掉重复的,低分的,选出正确的
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) # 在第二维增加一维 [1,22743,1]
    # 注意: (a[:,:,:]>5).float() 是再挑出a里面所有大于5的置为1, 其余的小于5的置为0

    prediction = prediction * conf_mask # [1,22743,85]
    # 这步结束后,所有刚才因为小于confidence而变成0的,现在它整个bbox的信息全为0了!!!(类似操作后面很常见: 先限定一下(不符合限定的置为0),再乘以自己本身--去掉不符合某一维度限定的数据)

    # transform the (center x, center y, height, width) attributes of our boxes, to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2) # 左上角x坐标 = xcenter-(hight/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4] # 再把prediction的[:,:,0和1和2和3]都变成刚才求出来的box_corner

    # 下面进行非极大值抑制 NMS, 但是由于每张图片的confidence不一样,所以要把batch拆分成一张一张图片进行(使用for循环)
    batch_size = prediction.size(0)
    write = False
    for ind in range(batch_size):
        image_pred = prediction[ind] #一张图的tensor
        # 现在每一bbox的行里面有85个值,但其中80个是代表80个类的分数(没必要这么存储,浪费空间),因为一个bbox中的obj只能属于一个类,我们只关注分数最高的那个类就好了,因此我们删掉80个,并增加一个索引到最高分的类以及该类的分数
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)
        # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引(返回最大元素在这一行的列索引)
        # 这个函数非常完美的符合我们的需求!!直接上它

        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf,max_conf_score) # 是个tuple
        image_pred = torch.cat(seq,1) # [22743,7], 7的来源是5+2(也就是上一步的tuple),5是:bbox坐标+该bbox有物体的概率,2是:物体得分,物体类别

        non_zero_ind = (torch.nonzero(image_pred[:,4])) # [5,1], 索引到具有非零conf的bbox位置. 5表示有几个物体检测出来
        try: # 这里的try是为了解决: 一个物体也没有检测出来的情况
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7) # 从image_pred里面取出这些bbox的信息 [5,7]
        except: # 如果no detections, 跳出循环体
            continue

        if image_pred_.shape[0] == 0:
            continue
        # 获得各个class
        img_classes = unique(image_pred_[:,-1]) # 将同一张图中多个相同的类合并,为了统计出该图包含的类别数目,以便进行NMS

        for cls in img_classes:
            # 对每个类别的bbox分别处理
            #  NMS
            # 首先提取这个类的所有detection
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1) # 挑出image_pred_[:,-1] == cls的置为1,其余置为0,扩增维度,再重新乘上自己
                                                                                #这波操作猛如虎,实现了: 过滤掉非本类的所有bbox,保留本类的bbox的所有信息
            cls_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze() # [2,3,4] : 表明了哪些位置是属于本类的
            image_pred_class = image_pred_[cls_mask_ind].view(-1,7) # 把这三个位置[2,3,4]的所有信息从image_pred_提取出来(也就是把0的全部删掉)

            # 按照confidence的顺序降序排序
            conf_sort_index = torch.sort(image_pred_class[:,4],descending = True)[1] # [2,1,0]: 只是排出序号, ### 按照bbox objectiveness confidence大小排序!按照存在物体的可能性大小排序
            image_pred_class = image_pred_class[conf_sort_index] # 按照序号,改变image_pred_class顺序
            idx = image_pred_class.size(0) # number of detection

            for i in range(idx):
                # 首先,一起出现在某一个range里的,每次都是同样的类别
                # 第一个出现的i = 0 的,是该类别得分最高的, 往后依次递减
                # 计算当前这个第i号bbox和 其他所有bbox的 IOU
                # IOU大于阈值,则把置信度低的删掉
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:]) # 分别计算当前bbox与后面所有bbox的iou
                    # 计算后的ious是多行tensor,包含第i个bbox和第[i+1:]后面所有bbox的iou值
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1) # 挑出ious < nms_conf的置为1, 其余置为0.
                                                                  # ious < nms_conf 代表:两者是同一类别,但非同一物体, 因此要留下,置为1 !!
                image_pred_class[i+1:] *= iou_mask  # [i+1:]表示,当前bbox后面的所有box. 乘上原来的自己, 把刚置为0的那些bbox的全部信息都置为0

                # Remove the non-zero entries 我觉得这里写错了,应该是removel zero entry
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()  # 返回非零元素的索引
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7) # 根据索引,从image_pred_class中挑出这些bbox及相关信息(7列)
                # 这步之后的结果,首先 i 之前的元素是在的(或者在之前已经被处理完了)
                # i 之后的元素(bbox) ,如果和当前的bbox重合太多,就删掉; 重合不多,就留下
            # 循环结束的时候,表示本类别(比如:person)的所有bbox的NMS已经进行完毕
        # 该循环结束表示所有类别的NMS进行完毕

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)  # 这里把每个batch的batch_id都加进来,是为了以后要用哦!
            seq = batch_ind, image_pred_class # 合并batch_ind和image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    # 因为有可能一个detection也没有,output也没有初始化,只能返回0了
    try:
        return output
    except:
        return 0


def unique(tensor):
    # tensor是一个长度为[5]的一维矩阵: [7,2,0,0,0]
    tensor_np = tensor.cpu().numpy() # 把tensor先变成numpy
    unique_np = np.unique(tensor_np) # 使用numpy方法unique,去除数组中的重复数字，并进行排序之后输出。
    unique_tensor = torch.from_numpy(unique_np) # 再从numpy变回来: [0,2,7]
    tensor_res = tensor.new(unique_tensor.shape) # [0,2,7]
    tensor_res.copy_(unique_tensor) # [0,2,7] 不明白为啥要新建之后再复制一下?--回答:内容不变,但复制一下就变成了cuda型!!!
    return tensor_res





