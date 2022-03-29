# Flexible integration for any Python script
import wandb  #67dc7c583abb77f32c739eba81f9f85a82b55b9a

#如果是第一次使用，需要在https://wandb.ai注册，然后在终端输入wandb login 按照提示登录到本电脑
#脚本执行完毕 wandb会自动将记录结果上传至网上可供查看分享，如：https://wandb.ai/jedi008/study_wandb/runs/32sm1v8u

# 1. Start a W&B run
wandb.init(project='study_wandb')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training here
# ‍3. Log metrics over time to visualize performance
loss = 0.9527
wandb.log({"loss": loss})