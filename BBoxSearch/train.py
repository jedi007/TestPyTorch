from samplest_yolov3_gray import *
from ImagesAndLabelsSet import *
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--data', type=str, default='D:/TestData/data/my_data.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='D:/TestData/weights/yolov3spp-voc-512.pt',
                        help='initial weights path')

    opt = parser.parse_args()

    train_path = "D:/TestData/data/my_train_data.txt"


    train_dataset = ImagesAndLabelsSet(train_path, 512, batch_size=4 )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=4,
                                                   num_workers=1,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle= False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    for i, (imgs, targets, paths, _, _) in enumerate(train_dataloader):
        print("targets: ",targets)
        for i in range(4):
            target = targets[i]
            print("target: ",target)



            img_o = imgs[i]

            img_numpy = img_o.permute(1, 2, 0).numpy()[:,:,[2,1,0]]
            print( type(img_numpy) )
            print( img_numpy.shape )

            img_gray = cv2.cvtColor(img_numpy,cv2.COLOR_BGR2GRAY)

            cv2.imshow("im_gray",img_gray) 

            model = YOLOv3Model_Gray()
            device = torch.device("cuda:0")
            model.to(device)
            model.train()

            input = torch.tensor(img_gray.reshape(1,1,512,512),device=device)/255.
            pred = model( input )
            print( "pred.shape: ",pred.shape )
            print( "pred: ",pred )

        

            target = targets[targets[:,0]==i]

            bboxes = target[:, 2:].detach().cpu().numpy()*img_o.shape[1]
            bboxes = xywh2xyxy(bboxes)

            scores = torch.ones_like(target[:,1]).cpu().numpy()
            classes = target[:, 1].detach().cpu().numpy().astype(np.int) + 1

            basepath = "D:/TestData"
            weights = opt.weights  # 改成自己训练好的权重文件
            json_path = basepath+"/data/pascal_voc_classes.json"  # json标签文件
            assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
            json_file = open(json_path, 'r')
            class_dict = json.load(json_file)
            category_index = {v: k for k, v in class_dict.items()}

            img_o = img_o.permute(1, 2, 0).numpy()[:,:,[2,1,0]]
            img_o = draw_box( img_o[:, :, ::-1], bboxes, classes, scores, category_index)

            img_o = np.array(img_o)
            cv2.imshow('detection', img_o)
            key = cv2.waitKey(3000000)
            if(key & 0xFF == ord('q')):
                break
        break
