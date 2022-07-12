import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import copy
import os

from nets.deeplabV3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

# model.eval() 和 with torch.no_grad() -> 如下
# https://www.bilibili.com/read/cv14752764  eval()
# https://blog.csdn.net/weixin_40548136/article/details/105122989

def main():
    model_path = 'logs/last_epoch_weights.pth'
    num_classes = 2
    HEIGHT = 512
    WIDTH = 512
    # ---------------------------------------------------#
    #   定义了输入图片的颜色，当我们想要去区分两类的时候
    #   我们定义了两个颜色，分别用于背景和斑马线, 在输出像素点上加上class_colors, 所以[0,0,0]并不是代表黑色
    #   [0,0,0], [0,255,0]代表了颜色的RGB色彩
    # ---------------------------------------------------#
    class_colors = [[0, 0, 0], [0, 255, 0]]

    # 加载模型
    model = DeepLab(num_classes=num_classes, backbone="mobilenet", downsample_factor=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    # 使用 GPU 进行计算
    model = model.cuda()
    print('{} model, and classes loaded.'.format(model_path))

    # --------------------------------------------------#
    #   对imgs文件夹进行一个遍历
    # --------------------------------------------------#
    imgs = os.listdir("img/")
    for jpg in imgs:
        img = Image.open("img/" + jpg)
        # 进行图片格式检查,若不是3通道RGB, 则转换
        img = cvtColor(img)
        old_img = copy.deepcopy(img)
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]
        # --------------------------------------------------#
        #   对输入进来的每一个图片进行Resize
        #   resize成[HEIGHT, WIDTH, 3], 下面的resize_image为不
        #   失真reize, 补灰条(也可以直接resize)
        #   image.new 的 size -> (w, h)
        #   resize    的 size -> (w, h)
        #   reshape   的 size -> (h, w)
        #   image.shape ->  [h, w, 3]
        # --------------------------------------------------#
        image_data, nw, nh = resize_image(img, (WIDTH, HEIGHT))
        # 通过np.transpose 进行维度交换  jpg[h][w][3] = jpg[3][h][w] 转成pytorch的张量形式 的numpy
        # 通过 expand_dims 进行增维, 符合送入检测格式
        # preprocess_input  已经归一化 / 255
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # --------------------------------------------------#
        #   将图像输入到网络当中进行预测
        # --------------------------------------------------#
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda()
            pr = model(images)[0]

            # prr = pr.cpu().numpy()
            # np.savetxt("img/pr1_1.txt", prr[0], fmt='%f', delimiter=' ')
            # np.savetxt("img/pr1_2.txt", prr[1], fmt='%f', delimiter=' ')

            # pr.permute 维度变换 [2, 512, 512] -> [512, 512, 2]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # np.savetxt("img/pr2_1.txt", pr[:, :, 0], fmt='%f', delimiter=' ')
            # np.savetxt("img/pr2_2.txt", pr[:, :, 1], fmt='%f', delimiter=' ')

            #   将灰条部分截取掉
            pr = pr[int((HEIGHT - nh) // 2): int((HEIGHT - nh) // 2 + nh), int((WIDTH - nw) // 2): int((WIDTH - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            # 经过 softmax 后 [512,512,2] 中的第三维中的2个值，被映射到0 - 0.999999之间
            # 通过 argmax 输出 第三维中的2个值较大值的索引 [512,512,2] -> [512, 512]
            pr = pr.argmax(axis=-1)
            # np.savetxt("img/pr3_1.txt", pr, fmt='%f', delimiter=' ')

            seg_img = np.zeros((orininal_h, orininal_w, 3))
            for c in range(num_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c) * class_colors[c][0]).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * class_colors[c][1]).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * class_colors[c][2]).astype('uint8')

            # 掩膜生成
            seg_img = Image.fromarray(np.uint8(seg_img))

            #   将新图片和原图片混合  0.3为 seg_img 透明度
            image = Image.blend(old_img, seg_img, 0.3)
            image.save("img_out/" + jpg)


if __name__ == "__main__":
    main()