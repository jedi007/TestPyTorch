# 获取CCPD数据集中的牌照信息

import random
import os
import cv2
import numpy as np

rovincelist = [ "皖", "沪", "津", "渝", "冀",
                "晋", "蒙", "辽", "吉", "黑",
                "苏", "浙", "京", "闽", "赣",
                "鲁", "豫", "鄂", "湘", "粤",
                "桂", "琼", "川", "贵", "云",
                "藏", "陕", "甘", "青", "宁",
                "新"]

wordlist = ["A", "B", "C", "D", "E",
            "F", "G", "H", "J", "K",
            "L", "M", "N", "P", "Q",
            "R", "S", "T", "U", "V",
            "W", "X", "Y", "Z", "0",
            "1", "2", "3", "4", "5",
            "6", "7", "8", "9"]


if __name__ == '__main__':
    random.seed(0)

    note_train = open('train.txt',mode='w')
    for root,dirs,files in os.walk("./CCPDdata/CCPD2020/ccpd_green/train"): 
        for file in files: 
            print("file: ", file)
            pathname = os.path.join(root,file).replace("\\","/")
            
            if file.endswith(".jpg"):
                name_list = file.split("-")
                # print("name_list: ", name_list) #['00947482638889', '95_272', '318&413_466&478', '462&478_318&461_320&413_466&428', '0_0_3_24_30_25_29_25', '170', '41.jpg']
                
                if len(name_list) != 7:
                    continue

                lp = name_list[4].split("_")
                lp = list(map(int, lp))
                lp_str = rovincelist[lp[0]]

                for i in range(1, len(lp)):
                    lp_str += wordlist[lp[i]]

                note_train.write(f"{lp_str}.jpg\t{lp_str}\n")


                points = name_list[3].split("_")
                points = [points[2], points[3], points[0], points[1]] # 变为从左上角开始顺时针排序的点位
                points = [list(map(int, i.split('&'))) for i in points]

                left = min(points[0][0], points[3][0])
                right = max(points[1][0], points[2][0])
                top = min(points[0][1], points[1][1])
                bottom = max(points[2][1], points[3][1])

                img = cv2.imdecode(np.fromfile(pathname, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert img is not None, 'Image Not Found ' + pathname

                license_plate = img[top:bottom, left:right]
                # cv2.imshow("license_plate show", license_plate)
                # cv2.waitKey() 
                # cv2.imwrite(f'./output/images/{lp_str}.jpg',license_plate)  #中文文件名乱码
                cv2.imencode('.jpg', license_plate)[1].tofile(f'./output/images/{lp_str}.jpg') 
                



                # exit(0)
                