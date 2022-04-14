import paddle
print("paddle " + paddle.__version__)

x_data = paddle.to_tensor([[1.], [3.0], [5.0], [9.0], [10.0], [20.0]])
y_data = paddle.to_tensor([[12.], [16.0], [20.0], [28.0], [30.0], [50.0]])

linear = paddle.nn.Linear(in_features=1, out_features=1)

w_before_opt = linear.weight.numpy().item()
b_before_opt = linear.bias.numpy().item()

print("w before optimize: {}".format(w_before_opt))
print("b before optimize: {}".format(b_before_opt))


mse_loss = paddle.nn.MSELoss()
sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters = linear.parameters())

total_epoch = 10000
for i in range(total_epoch):
    y_predict = linear(x_data)
    loss = mse_loss(y_predict, y_data)
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_grad()
    
    if i%1000 == 0:
        print("epoch {} loss {}".format(i, loss.numpy()))
        
print("finished training， loss {}".format(loss.numpy()))


w_after_opt = linear.weight.numpy().item()
b_after_opt = linear.bias.numpy().item()

print("w after optimize: {}".format(w_after_opt))
print("b after optimize: {}".format(b_after_opt))