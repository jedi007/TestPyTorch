from samplest_yolov3_gray import *
from ImagesAndLabelsSet import *
import cv2
from build_utils.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--data', type=str, default='D:/TestData/data/my_data.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='D:/TestData/weights/yolov3spp-voc-512.pt',
                        help='initial weights path')


    opt = parser.parse_args()
    device = torch.device("cuda:0")

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
        print("imgs.shape: ",imgs.shape)
        
        inputs = torch.zeros( (imgs.shape[0],1,imgs.shape[2],imgs.shape[3]),device=device )
        for i in range( imgs.shape[0] ):
            img_o = imgs[i]

            img_numpy = img_o.permute(1, 2, 0).numpy()[:,:,[2,1,0]]
            img_gray = cv2.cvtColor(img_numpy,cv2.COLOR_BGR2GRAY)

            inputs[i] = torch.tensor(img_gray.reshape(1,1,512,512),device=device)/255.
            


        model = YOLOv3Model_Gray()
        model.to(device)
        model.train()

        print("inputs.shape: ",inputs.shape)
        pred = model( inputs )
        print( "pred.shape: ",pred.shape )
        #print( "pred: ",pred )

        test = torch.tensor(pred[0].shape)[[3, 2, 3, 2]] 
        print("test: ",test.shape)
        print("test: ",test)



        loss = compute_loss(pred, targets, model)
        print("loss: ",loss)


        


        exit(0)