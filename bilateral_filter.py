import numpy as np
import cv2
import math
from multiprocessing import Pool
import time


# 开启的多进程数目
PROCESS_NUM = 3
# 滤波器大小
FILTER_DIAMETER = 7
# 空间域σ值
SIGMA_D = 10
# 值域σ值
SIGMA_R = 30


# 高斯函数计算
def gaussian(x, sigma):
    return (1 / (2 * math.pi * (sigma ** 2))) * math.exp(-(x ** 2) / (2 * (sigma ** 2)))

# 提前计算空间域数组
def get_space_Array(diameter, sigma_d):
    space = []
    k = int(diameter / 2)
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            space.append(gaussian(np.sqrt(i ** 2 + j ** 2), sigma_d))
    return space

# 提前计算值域数组
def get_color_Array(sigma_r):
    color = []
    for i in range(0, 256):
        color.append(gaussian(i, sigma_r))
    return color

# 双边滤波处理
def bilateral_filter(source, filtered_image, row, col, diameter, space ,color):
    # 滤波器大小的一半
    half_dia = int(diameter / 2)
    i_filtered = 0
    wp = 0
    for i in range(0, diameter):
        neighbour_row = int(row - (half_dia - i))
        if 0 <= neighbour_row < len(source):
            for j in range(0, diameter):
                neighbour_col = int(col - (half_dia - j))
                if 0 <= neighbour_col < len(source[0]):
                    color_temp = abs(int(source[row][col]) - int(source[neighbour_row][neighbour_col]))
                    gauss_r = color[color_temp]
                    gauss_d = space[i * diameter + j]
                    w = gauss_r * gauss_d
                    i_filtered += w * source[neighbour_row][neighbour_col]
                    wp += w
    i_filtered = i_filtered / wp
    filtered_image[row][col] = int(i_filtered)

# 遍历整张图片，对每个像素点进行双边滤波
def bilateral_filter_process(args):
    source, filter_diameter, sigma_r, sigma_d = args
    filtered_image = np.zeros(source.shape,dtype="uint8")
    space = get_space_Array(filter_diameter, sigma_d)
    color = get_color_Array(sigma_r)
    for i in range(0,len(source)):
        for j in range(0,len(source[0])):
            bilateral_filter(source, filtered_image, i, j, filter_diameter, space, color)
    return filtered_image


if __name__ == "__main__":
    src = cv2.imread("src.jpg")
    cv2.imshow("src",src)

    t1 = time.time()
    if len(src.shape) != 3:
        dst = bilateral_filter_process(args=[src, FILTER_DIAMETER, SIGMA_R, SIGMA_D])
    else:
        blue, green, red = cv2.split(src)
        pool = Pool(processes=PROCESS_NUM)
        # 多进程处理
        future_blue, future_green, future_red = pool.map(bilateral_filter_process,
                                                         [(blue, FILTER_DIAMETER, SIGMA_R, SIGMA_D),
                                                          (green, FILTER_DIAMETER, SIGMA_R, SIGMA_D),
                                                          (red, FILTER_DIAMETER, SIGMA_R, SIGMA_D)])
        pool.close()
        pool.join()
        # 三通道融合
        dst = cv2.merge([future_blue, future_green, future_red])  # merge back to one img
    t2 = time.time()

    print(t2 - t1)
    cv2.imshow("dst", dst)
    cv2.imwrite("dst.jpg", dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
