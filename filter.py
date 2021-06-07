import math
import cv2
import numpy as np


def padPrpcess(img, rows, cols):
    P = 2 * rows
    Q = 2 * cols
    pimg = np.zeros((P, Q), dtype="uint8")
    pimg[:rows, :cols] = img
    return pimg, P, Q


def centerProcess(img, rows, cols):
    cimg = np.copy(img)
    cimg = cimg.astype(np.float32)
    for i in range(rows):
        for j in range(cols):
            cimg[i, j] = img[i, j] * (math.pow(-1, i + j))
    return cimg


def magnitudeSpect(DFTImg):
    # 取绝对值：将复数变化成实数
    # 取对数的目的为了将数据变化到较小的范围（比如0-255）
    magDFTImg = np.abs(DFTImg)
    magDFTImg = 20 * np.log(magDFTImg+0.0000001)
    magDFTImg = magDFTImg.astype(np.int8)
    return magDFTImg


def sImg(img):
    simg = np.abs(img)
    simg = simg.astype(np.int8)
    return simg


def gaussianFrequencyFilter(img, rows, cols, sigma=20):
    centerx = rows // 2
    centery = cols // 2
    for i in range(rows):
        for j in range(cols):
            img[i, j] *= np.exp(-((i - centerx) ** 2 + (j - centery) ** 2) / (2 * sigma ** 2))


def process(srcimg):
    # 读取灰度图
    img = cv2.imread(srcimg, 0)
    cv2.imshow("src", img)

    rows, cols = img.shape[:2]

    # 填充
    pimg, P, Q = padPrpcess(img, rows, cols)
    # cv2.imshow("pimg", pimg)

    # 中心化
    cimg = centerProcess(pimg, P, Q)
    # cv2.imshow("cimg", sImg(cimg))

    # 傅里叶变换
    fft = np.fft.fft2(cimg)
    # cv2.imshow("fft", magnitudeSpect(fft))

    # 高斯低通滤波器
    gaussianFrequencyFilter(fft, P, Q, 200)
    cv2.imshow("pfft", magnitudeSpect(fft))

    # 逆傅里叶变换
    ifft = np.fft.ifft2(fft)
    # 取实部
    ifft = np.real(ifft)
    res = centerProcess(ifft, P, Q)
    res = res.astype(np.uint8)
    # cv2.imshow("res", res)
    dst = np.zeros((rows, cols), dtype="uint8")
    for i in range(rows):
        for j in range(cols):
            dst[i, j] = res[i, j]
    cv2.imshow("dst", dst)
    cv2.waitKey(0)


if __name__ == "__main__":
    src = "src1.jpg"
    process(src)
