import os

models = ['GraphSAGE']  ##
test_time = 2 ##
learning_rates = [1e-3] ##
epochs = [300]
batch_size = [32] ##

num_neighbors = [5,10] ##
node_hidden_dim = [32,64] ##
num_layers = [1, 2] ##

num_lstm_layers = [1]
num_cpu = 16

gpu = 1

log_dir = 'experiment'
data_dir = '../graph_transfer_learning-main/data/small_phish/'
for t in range(test_time):
    for m in models:
        for l in num_layers:
            for n in num_neighbors:
                for d in node_hidden_dim:
                    for bs in batch_size:
                        for lr in learning_rates:
                            for e in epochs:
                                log_name = "{model}_{num_layers}_{num_neighbors}_{node_hidden_dim}_{batch_size}_{lr}_{num_heads}_{test_time}_{epochs}".format(
                                        model=m, num_layers=l, num_neighbors=n, node_hidden_dim=d, batch_size=bs, lr=lr, num_heads=1, test_time=t, epochs=e)
                                print(log_name)
                                os.system(
                                        "python -u main.py --data_dir {data_dir} "
                                        "--model {model} --gpu {gpu} --epochs {epochs} "
                                        "--num_neighbors {num_neighbors} --node_hidden_dim {node_hidden_dim} "
                                        "--num_layers {num_layers} --num_heads {num_heads} --num_cpu {num_cpu} "
                                        "--log_dir {log_dir} --log_name {log_name} --num_lstm_layers {num_lstm_layers} "
                                        "--batch_size {batch_size} --lr {lr}".format(data_dir=data_dir, model=m, 
                                                                                           epochs=e, gpu=gpu, num_neighbors=n,
                                                                                           node_hidden_dim=d, num_layers=l, num_heads=1,
                                                                                           num_cpu=num_cpu, log_dir=log_dir,log_name=log_name, 
                                                                                           num_lstm_layers=1, batch_size=bs, lr=lr))



'''
models=['GTEA-LSTM', 'GTEA-LSTM+T2V', 'GTEA-Trans', 'GTEA-Trans+T2V']
for t in range(test_time):
    for lstm_l in num_lstm_layers:
        for m in models:
            for l in num_layers:
                for n in num_neighbors:
                    for d in node_hidden_dim:
                        for bs in batch_size:
                            for lr in learning_rates:
                                for e in epochs:
                                    log_name = "{model}_{num_layers}_{num_neighbors}_{node_hidden_dim}_{num_lstm_layers}_{batch_size}_{lr}_{test_time}_{epochs}".format(model=m, num_layers=l, num_neighbors=n, node_hidden_dim=d, num_lstm_layers=lstm_l, batch_size=bs, lr=lr, test_time=t, epochs=e)
                                    print(log_name)
                                    os.system(
                                                "python -u main.py --data_dir {data_dir} "
                                                "--model {model} --gpu {gpu} --epochs {epochs} "
                                                "--num_neighbors {num_neighbors} --node_hidden_dim {node_hidden_dim} "
                                                "--num_layers {num_layers} --num_workers {num_workers} "
                                                "--log_dir {log_dir} --log_name {log_name} --num_lstm_layers {num_lstm_layers} "
                                                "--batch_size {batch_size}  --lr {lr}  --time_hidden_dim 4".format(data_dir=data_dir, model=m,
                                                                                                   epochs=e, gpu=gpu, num_neighbors=n,
                                                                                                   node_hidden_dim=d, num_layers=l, 
                                                                                                   num_workers=num_workers,log_dir=log_dir,log_name=log_name,
                                                                                                   num_lstm_layers=lstm_l, batch_size=bs, lr=lr))
'''
