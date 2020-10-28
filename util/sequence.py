import imgaug as ia
from imgaug import augmenters as iaa


def get():
    def sometimes(aug): return iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            # apply the following augmenters to most images
            # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width

            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),

            sometimes(iaa.Affine(
                # scale images to 90-110% of their size, individually per axis
                scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                # translate by -10 to +10 percent (per axis)
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                rotate=(-20, 20),  # rotate by -20 to +20 degrees
                # shear=(-16, 16),  # shear by -16 to +16 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                mode=ia.ALL
            )),

            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),
                       [
                           # convert images into their superpixel representation
                           sometimes(iaa.Superpixels(
                               p_replace=(0, 1.0), n_segments=(20, 200))),
                           iaa.OneOf([
                               # blur images with a sigma between 0 and 3.0
                               # 使用0和1.5之間的sigma模糊圖像
                               iaa.GaussianBlur((0, 1.5)),
                               # blur image using local means with kernel sizes between 2 and 7
                               # 使用局部大小在2到5之間的內核來模糊圖像
                               iaa.AverageBlur(k=(2, 5)),
                               # blur image using local medians with kernel sizes between 2 and 7
                               # 使用內核大小在2到7之間的局部中值模糊圖像
                               # iaa.MedianBlur(k=(2, 7)),
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(
                               0.75, 1.5)),  # sharpen images 銳化圖像

                           # iaa.Emboss(alpha=(0, 1.0), strength=(
                           #    0, 2.0)),  # emboss images 浮雕圖像

                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           # 搜索所有邊緣或有向邊緣，使用過大的蒙版將結果與原始圖像融合

                           # add gaussian noise to images
                           # 給圖像增加高斯噪聲
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.3),
                           iaa.OneOf([
                               # randomly remove up to 10% of the pixels
                               # 隨機刪除多達10％的像素
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(
                                   0.02, 0.05), per_channel=0.2),
                           ]),
                           # invert color channels
                           # 反轉顏色通道
                           # iaa.Invert(0.05, per_channel=True),
                           # change brightness of images (by -10 to 10 of original value)
                           # 更改圖像的亮度（原始值的-10到10）
                           iaa.Add((-5, 5), per_channel=0.5),
                           # change hue and saturation
                           # 改變色調和飽和度
                           iaa.AddToHueAndSaturation((-10, 10)),
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           # 更改整個圖像的亮度（有時每通道）或更改子區域的亮度
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-4, 0),
                                   first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   second=iaa.ContrastNormalization((0.5, 2.0))
                               )
                           ]),
                           # improve or worsen the contrast
                           # 提升或降低對比度
                           iaa.ContrastNormalization((0.5, 1.0), per_channel=0.5),
                           iaa.Grayscale(alpha=(0.0, 0.5)),
                           # move pixels locally around (with random strengths)
                           # 局部移動像素（具有隨機強度）
                           sometimes(iaa.ElasticTransformation(
                               alpha=(0.5, 3.5), sigma=0.25)),
                           # sometimes move parts of the image around
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )

        ],
        random_order=True
    )
