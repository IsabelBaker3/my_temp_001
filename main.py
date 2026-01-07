# -*- coding: utf-8 -*-

import os
import cv2
import glob
import utils

my_COD_image_root = r'H:\mbzuai\COD\ckpt_model_hz_4_1_m198_all\COD10K\*.png'
baseline_COD_image_root = r'E:\2-data-CamMap\0-myModel\ckpt_model_hz_baseline_m40\COD10K\*.png'
rgb_image_root = r'E:\Datasets\COD\COD_dataset\TestDataset\COD10K\Imgs\*.jpg'

my_COD_heatmap_save_root = r'E:\2-data-CamMap\0-myModel\heatmap\my_m198'
baseline_COD_heatmap_save_root = r'E:\2-data-CamMap\0-myModel\heatmap\baseline40'

for my_COD_image_file, baseline_COD_image_file, rgb_image_file in zip(glob.glob(my_COD_image_root),
                                                                      glob.glob(baseline_COD_image_root),
                                                                      glob.glob(rgb_image_root)):
    # imgs = []
    rgb_image = cv2.imread(rgb_image_file, flags=1)
    image_name = my_COD_image_file.split('\\')[-1][0:-4]
    # 读取检测结果
    my_cod_img = cv2.imread(my_COD_image_file)
    my_cod_img_gray = cv2.cvtColor(my_cod_img, cv2.COLOR_BGR2GRAY)
    # 读取baseline结果
    baseline_cod_img = cv2.imread(baseline_COD_image_file)
    baseline_cod_img_gray = cv2.cvtColor(baseline_cod_img, cv2.COLOR_BGR2GRAY)

    # 转热图
    my_heatmap_img = cv2.applyColorMap(my_cod_img, cv2.COLORMAP_JET)
    baseline_heatmap_img = cv2.applyColorMap(baseline_cod_img_gray, cv2.COLORMAP_JET)

    # 叠加原图
    my_heatmap_img_add=cv2.addWeighted(rgb_image,0.3,my_heatmap_img,0.7,0)
    baseline_img_add=cv2.addWeighted(rgb_image,0.3,baseline_heatmap_img,0.7,0)

    # cv2.imshow("imagezz", rgb_image)
    # 保存my结果
    my_COD_save_image_path = os.path.join(my_COD_heatmap_save_root, image_name + '.jpg')
    cv2.imwrite(my_COD_save_image_path, my_heatmap_img_add)
    # 保存baseline结果
    baseline_COD_save_image_path = os.path.join(baseline_COD_heatmap_save_root, image_name + '.jpg')
    cv2.imwrite(baseline_COD_save_image_path, baseline_img_add)

    # cv2.waitKey(0)
