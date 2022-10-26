# CCPD 转yolo格式

import random
import os


if __name__ == '__main__':
    random.seed(0)

    note_val = open('val.txt',mode='w')
    for root,dirs,files in os.walk("./CCPDdata/CCPD2020/ccpd_green/val"): 
        for file in files: 
            print("file: ", file)
            pathname = os.path.join(root,file).replace("\\","/")
            note_val.write(pathname+"\n")
            l = [pathname]
            if file.endswith(".jpg"):
                name_list = file.split("-")
                # print("name_list: ", name_list) #['00947482638889', '95_272', '318&413_466&478', '462&478_318&461_320&413_466&428', '0_0_3_24_30_25_29_25', '170', '41.jpg']
                
                if len(name_list) != 7:
                    continue

                points = name_list[3].split("_")
                points = [points[2], points[3], points[0], points[1]] # 变为从左上角开始顺时针排序的点位
                points = [list(map(int, i.split('&'))) for i in points]
                print("points: ",points)

                left = min(points[0][0], points[3][0])
                right = max(points[1][0], points[2][0])
                top = min(points[0][1], points[1][1])
                bottom = max(points[2][1], points[3][1])
                x = ((right - left) / 2 + left) / 720
                y = ((bottom - top) / 2 + top)/ 1160
                w = (right - left) / 720
                h = (bottom - top) / 1160

                note = open('./CCPDdata/CCPD2020/ccpd_green/val_txt/{}.txt'.format(file[0:-4]),mode='w')
                note.write(" ".join(map(str, [0, x, y, w, h])))
                note.close()
                
    note_val.close()
