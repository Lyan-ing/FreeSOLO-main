import cv2 as cv
import numpy as np
import time


# 直方图均衡化
def histogram(img):
    name = "histogram"
    pixels = np.array(img)
    ori_shape = img.shape

    pixels = pixels.flatten()
    pixel_sum = []  # 存储灰度级出现次数
    pixel_P = []  # 存储灰度级出现概率
    pixel_change_rule = []  # 存储灰度变化规则列表

    M = np.max(pixels)
    for i in range(M + 1):
        pixel_sum.append(np.sum(pixels == i))
    pixels_num = np.sum(pixel_sum)
    for i in range(M + 1):
        P = pixel_sum[i] / pixels_num
        pixel_P.append(P)  # 获得每个灰度级出现的概率
        P_sum = np.sum(pixel_P)  # 前i个灰度级出现的总概率
        pixel_change = int(round((M - 1) * P_sum))
        pixel_change_rule.append(pixel_change)  # 灰度对应规则列表

    # 改变灰度级
    L = len(pixel_change_rule)
    pixels_copy = pixels.copy()
    for i in range(L):
        pixels_copy[pixels == i] = pixel_change_rule[i]

    pixels_copy = pixels_copy.reshape(ori_shape)
    return pixels_copy, name


# 均值滤波
def average_filter(img):
    name = "average_filter"
    pixels = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    pixels_copy = pixels.copy()
    s_9 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    s_16 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    s_9_sum = (s_9).sum()
    s_16_sum = (s_16).sum()
    for i in range(height):
        for j in range(width):
            if (i > 0 and i < (height - 1)) and (j > 0 and j < (width - 1)):
                mask = pixels[i - 1:i + 2, j - 1:j + 2]
                fs_9 = (s_9) * mask
                fs_16 = (s_16) * mask
                filter_9 = ((fs_9).sum()) / s_9_sum
                filter_16 = ((fs_16).sum()) / s_16_sum
                pixels_copy[i][j] = np.clip(filter_9, 0, 255)

    return pixels_copy, name


# 中值滤波
def median_filter(img):
    name = "median_filter"
    pixels = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    pixels_copy = pixels.copy()
    for i in range(height):
        for j in range(width):
            if (i > 0 and i < (height - 1)) and (j > 0 and j < (width - 1)):
                mask = pixels[i - 1:i + 2, j - 1:j + 2]
                pixels_copy[i][j] = np.median(mask)

    return pixels_copy, name


# 二阶微分锐化
# 拉普拉斯算子
def laplace(img):
    name = "laplace"
    pixels = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    pixels_copy = pixels.copy()
    for i in range(height):
        for j in range(width):
            if (i > 0 and i < (height - 1)) and (j > 0 and j < (width - 1)):
                mask = pixels[i - 1:i + 2, j - 1:j + 2]
                #                 copy_1=9*int(mask[1][1])-(int(mask[1][0])+int(mask[0][1])+int(mask[1][2])
                #                                           +int(mask[0][0])+int(mask[0][2])+int(mask[2][0])+int(mask[2][1])+int(mask[2][2]))
                copy_1 = 5 * int(mask[1][1]) - (int(mask[1][0]) + int(mask[0][1]) + int(mask[1][2]) + int(mask[2][1]))
                pixels_copy[i][j] = np.clip(copy_1, 0, 255)

    return pixels_copy, name


# 一阶微分锐化

# Roberts交叉梯度算子
def roberts(img):
    name = "roberts"
    pixels = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    pixels_copy = pixels.copy()
    for i in range(height):
        for j in range(width):
            if (i < (height - 1) and j < (width - 1)):
                mask = pixels[i:i + 2, j:j + 2]
                gradient = abs(int(mask[1][1]) - int(mask[0][0])) + abs(int(mask[0][1]) - int(mask[1][0]))
                pixels_copy[i][j] = np.clip(gradient, 0, 255)

    return pixels_copy, name


# Prewitt算子
def prewitt(img):
    name = "prewitt"
    pixels = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    pixels_copy = pixels.copy()
    sx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    sy = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    for i in range(height):
        for j in range(width):
            if (i > 0 and i < (height - 1)) and (j > 0 and j < (width - 1)):
                mask = pixels[i - 1:i + 2, j - 1:j + 2]
                fsx = sx * mask
                fsy = sy * mask
                gradient = abs(fsx.sum()) + abs(fsy.sum())
                pixels_copy[i][j] = np.clip(gradient, 0, 255)

    return pixels_copy, name


# Sobel算子
def sobel(img):
    name = "sobel"
    pixels = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    pixels_copy = pixels.copy()
    sx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    for i in range(height):
        for j in range(width):
            if (i > 0 and i < (height - 1)) and (j > 0 and j < (width - 1)):
                mask = pixels[i - 1:i + 2, j - 1:j + 2]
                fsx = sx * mask
                fsy = sy * mask
                gradient = abs(fsx.sum()) + abs(fsy.sum())
                pixels_copy[i][j] = np.clip(gradient, 0, 255)

    return pixels_copy, name


def my_laplace(img):
    name = "my_laplace"
    pic = np.array(img)
    pic_laplace = pic.copy()
    # 定义laplace滤波器
    kernel = np.asarray([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    kernel_size = 3
    # 遍历原图
    for i in range(pic.shape[0] - kernel_size + 1):
        # 计算中心点（并且更新中心点横坐标）
        center = [kernel_size // 2 + i, kernel_size // 2]
        for j in range(pic.shape[1] - kernel_size + 1):
            # 计算卷积
            r = np.sum(pic[i:i + kernel_size, j:j + kernel_size] * kernel)
            # 因为图象为uint8，小于0时要用0代替，不然会由256-r代替
            if r < 0:
                r = 0
            if r > 255:
                r = 255
            # 卷积值替换中间点坐标
            pic_laplace[center[0], center[1]] = np.clip(pic[center[0], center[1]] - r, 0, 255)
            # 更新中间点纵坐标
            center[1] += 1

    return pic_laplace, name


# methods | picture_name | weather write the output ,default=0
def change_img(my_methods, picture, write=0):
    # 计时工具
    start_time = time.time()
    ori = cv.imread(f"img_test/{picture}.jpg", 1)
    cv.imshow("img", ori)
    iim = cv.imread(f"img_test/{picture}.jpg", 0)
    pixels_data1, name1 = my_methods(iim)
    cv.imshow("img_changed1", pixels_data1)
    # pixels_data = None
    # for i in range(3):
    #     img = ori[:, :, i]
    #     pixels_data_i, name = my_methods(img)
    #     if pixels_data is None:
    #         pixels_data = np.expand_dims(pixels_data_i, axis=2)
    #     else:
    #         pixels_data = np.concatenate((pixels_data, np.expand_dims(pixels_data_i, 2)), axis=2)
    # # pixels_data.transpose(1, 2, 0)
    #
    # end_time = time.time()
    # times = end_time - start_time
    # print("the time of computing is :", times, "s")
    # if write == 1:
    #     cv.imwrite(f"img_test/{picture}_{name}.jpg", pixels_data)
    #
    # cv.imshow("img_changed", pixels_data)

    cv.waitKey(0)
    cv.destroyAllWindows()


# 显示效果
# methods | picture_name | weather write the output ,default=0
img = change_img(prewitt, "3", 0)
