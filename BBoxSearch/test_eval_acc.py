from torch.serialization import save
from samplest_yolov3_gray import *
from ImagesAndLabelsSet import *
import cv2
from build_utils.utils import *


from tqdm import tqdm

current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录
print("current_work_dir: ",current_work_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=6)

    opt = parser.parse_args()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    train_path = "D:/TestData/data/my_train_data.txt"
    test_path = "D:/TestData/data/my_train_data.txt"

    # 验证集的图像尺寸指定为img_size(512)
    val_dataset = ImagesAndLabelsSet(test_path, 512, batch_size=opt.batch_size )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=opt.batch_size,
                                                num_workers=1,
                                                # Shuffle=True unless rectangular training is used
                                                shuffle= False,
                                                pin_memory=True,
                                                collate_fn=val_dataset.collate_fn)


    model = YOLOv3Model_Gray()
    model.load_state_dict(torch.load("D:\Study\GitHub\TestPyTorch\BBoxSearch\savepath\savemodel-35-mloss=0.4508.pt"))
    model.to(device)
    model.eval()

    
    total_right_pbj = 0
    total_pobj = 0
    total_tobj = 0

    for i, (imgs, targets, paths, _, _) in enumerate(val_dataloader):
        print(">>>>>>>>>>>>>>> {} / {} : ".format(i, val_dataset.batch_number ) )
        print(imgs.shape)
    
        inputs = torch.zeros( (imgs.shape[0],1,imgs.shape[2],imgs.shape[3]),device=device )

        for img_index in range( imgs.shape[0] ):
            img_o = imgs[img_index]

            img_numpy = img_o.permute(1, 2, 0).numpy()[:,:,[2,1,0]]
            img_gray = cv2.cvtColor(img_numpy,cv2.COLOR_BGR2GRAY)

            inputs[img_index] = torch.tensor(img_gray.reshape(1,1,512,512),device=device)/255.

        pred = model( inputs )

        #print("val: pred.shape: ",pred.shape)

        pobj = pred[...,[0]]
        #print("pobj.shape: ",pobj.shape)

        #print("pobj[0][0][0]:　",pobj[0][0][0])

        pobj = (pobj > 0.95).int()
        #print("after　pobj[0][0][0]:　",pobj[0][0][0])

        psum = pobj.sum()
        #print("psum: ",psum.item())
        total_pobj += psum



        targets, tobj = build_targets(pred, targets, device)  # targets
        #print( "tobj[0][0][0]:　",tobj[0][0][0] )

        tsum = tobj.sum()
        #print("tsum: ",tsum.int().item())
        total_tobj += tsum


        right_pobj = ((tobj.int() + pobj) == 2).int()
        #print("right_pobj.shape: ",right_pobj.shape)
        #print( "right_pobj[0][0][0]:　",right_pobj[0][0][0] )
        #print("right_pobj.sum: ",right_pobj.sum())
        total_right_pbj += right_pobj.sum()




        #print("targets.shape: ",targets.shape)
        #print("tobj.shape: ",tobj.shape)

    
        print("total_right_pbj: ",total_right_pbj)
        print("total_pobj: ",total_pobj)
        print("total_tobj: ",total_tobj)
    