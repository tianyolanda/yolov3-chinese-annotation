# yolov3-中文注释
"How to Implement YOLO v3 Object Detector from Scratch" 教程的源码+[逐行中文注释]

- 代码来源: https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
- 文字教程:https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
- yolo v3完整代码(inference): https://github.com/ayooshkathuria/pytorch-yolo-v3


# 使用
 1. 下载权重文件到目录
 ```
 wget https://pjreddie.com/media/files/yolov3.weights 
```
 2. 运行detect.py
 ```
python detect.py
```

# 其他说明
注释中的维度数字是以608*608 大小的输入图像为例的

# 源码流程简要介绍
以detect.py,也就是main()开始 
1. 用户输入,传递一堆参数
2. 构建yolo模型结构
>  model = Darknet(args.cfgfile)
3. 加载权重(训练好的权重)
>  model.load_weights(args.weightsfile)
4. 根据用户输入,修改模型参数(比如输入图像的长宽)
5. 读取图像,存储原图大小(未来还原用),预处理图像(resize等)
6. 将所有图片叠起来变成一个batch,一起检测
7. 检测1: 首先将batch送入model中,前向传播,经过许多卷积层,最后输出一个特征图(B x 10647 x 85维) [以608*608的图为例]
> 每张图片都检测出10647个结果,B 是batch size(一共有多少张图片), 85 = 物体类别80 + (x,y,w,h) + 本bbox是否有物体的confidence
8. 检测2: 对每张图的10647个检测结果进行NMS,挑选出有效bbox框,存成一个tensor(实现该功能的函数叫write_results())
> 这里注意,由于每幅图的NMS的confidence阈值不一样,因此要把每一张图从batch中拆开单独过一遍for循环.

>  NMS原理--每幅图都关注:同类型里最高分的bbox,并以他为基准判断其他同类型的bbox是否与他重复,再决定是否删掉)

9. 对每张图的检测,统计时间和bbo数目,输出
10. 画框
> 之前resize的时候填充/拉抻过图片,这里首先裁掉画到原图像外部的bbox框

> 将bbox框映射到原图,画上框,并存储检测后的新图
11. 统计运算时间等数据
