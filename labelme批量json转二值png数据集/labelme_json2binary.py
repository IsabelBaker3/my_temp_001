"""
根据labelme标注的json和对应的原图，生成标注图的binary图
若原图无标注则生成无标注binary图
"""
from multiprocessing import Pool
from tqdm import trange

import os
import numpy as np
import json
import cv2


def json2binary(labelme_dataset_root,save_root):
    files = os.listdir(labelme_dataset_root)
    bar=trange(len(files))
    for (img_file_,_) in zip(files,bar):
        if img_file_.endswith('.jpg'):
            img_filename=labelme_dataset_root+img_file_
            json_filename=img_filename.replace('.jpg','.json')
            img_=cv2.imread(img_filename)
            img=img_.copy()
            height,width,_=np.array(img).shape
            dst = np.zeros((height, width, 3))
            if os.path.exists(json_filename):
                json_file = json.load(open(json_filename, "r", encoding="utf-8"))
                for multi in json_file["shapes"]:
                    points = np.array(multi["points"]).astype(np.int64)
                    if len(points)==2: #为矩形框标注
                        cv2.rectangle(dst,points[0],points[1],(255, 255, 255),thickness=-1)
                    else: # 为多边形标注
                        cv2.fillPoly(dst, [points], (255, 255, 255), cv2.LINE_AA)
                b, g, r = cv2.split(dst)
                r[np.where(r != 0)] = 255
                cv2.imwrite(os.path.join(save_root, img_file_.replace('.jpg', '.png')), r)
            else:
                b, g, r = cv2.split(dst)
                cv2.imwrite(os.path.join(save_root, img_file_.replace('.jpg', '.png')), r)
        else:
            pass


if __name__ =='__main__':
    labelme_dataset_root ="E:/Datasets/anomaly_detection/screw/handmade_annotation/screw_672/train/1/img/"
    save_root = "E:/Datasets/anomaly_detection/screw/handmade_annotation/screw_672/train/1/gt/"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    # with Pool(8) as P:
    #     list(tqdm.tqdm(P.imap(json2binary),total=len()))
    json2binary(labelme_dataset_root,save_root)
    print("finished!")