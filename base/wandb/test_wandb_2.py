#如果是第一次使用，需要在https://wandb.ai注册，然后在终端输入wandb login 按照提示登录到本电脑
#脚本执行完毕 wandb会自动将记录结果上传至网上可供查看分享，如：https://wandb.ai/jedi008/study_wandb/runs/32sm1v8u

# Inside my model training code
import wandb
wandb.init(project="study_wandb")

wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128

def my_train_loop():
    loss = 10
    accuracy = 10
    for epoch in range(100):
        loss -= epoch/100  #模拟真实训练中的变化
        accuracy += epoch/100  #模拟真实训练中的变化
        wandb.log({'epoch': epoch, 'Loss': loss, 'Accuracy':accuracy})  #在浏览器中按照曲线显示


def main():
    my_train_loop()

if __name__ == '__main__':
    main()

