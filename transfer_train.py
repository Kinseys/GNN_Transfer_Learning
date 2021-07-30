import argparse, time, math
import numpy as np
import sys
import logging
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from models import GraphSAGE
#from models import GTEALSTM

from miniBatchTrainer import MiniBatchTrainer

from data.dynamic_event_data_loader import Dataset
#from data.Tencent.DataGenerator import DataGenerator




def main(args):

    log_path = os.path.join(args.log_dir, args.model + time.strftime("_%m_%d_%H_%M_%S", time.localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logging.basicConfig(filename=os.path.join(log_path, 'log_file'),
                        filemode='w',
                        format='| %(asctime)s |\n%(message)s',
                        datefmt='%b %d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(args)

#    data_generator = DataGenerator(train_ratio = args.train_ratio, data_dir=args.data_dir)

    data = Dataset(data_dir=args.data_dir, max_event=args.max_event, use_K=args.use_K, K=args.K, remove_node_features=args.remove_node_features)

    g = data.g

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    # num_edge_feats = data.num_edge_feats

    train_small = int(len(labels)/10)
    train_mask = torch.BoolTensor(data.train_mask)
    train_mask = train_mask[:train_small]

    val_mask = torch.BoolTensor(data.val_mask)
    val_mask = val_mask[:train_small]

    test_mask = torch.BoolTensor(data.test_mask)
    test_mask = test_mask[:train_small]

    train_id = np.nonzero(data.train_mask)[0].astype(np.int64)
    val_id = np.nonzero(data.val_mask)[0].astype(np.int64)
    test_id = np.nonzero(data.test_mask)[0].astype(np.int64)
    print('...............')
    print(train_id)
    print('...............')


    num_nodes = features.shape[0]
    node_in_dim = features.shape[1]

    num_edges = data.g.number_of_edges()
    edge_in_dim = data.edge_in_dim
    edge_timestep_len = data.edge_timestep_len
    num_classes = data.num_classes
    

    num_train_samples = train_mask.int().sum().item()
    num_val_samples = val_mask.int().sum().item()
    num_test_samples = test_mask.int().sum().item()

    logging.info("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Node_feat %d
      #Edge_feat %d
      #Edge_timestep %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (num_nodes, num_edges, 
           node_in_dim, edge_in_dim, edge_timestep_len,
              num_classes,
              num_train_samples,
              num_val_samples,
              num_test_samples))
    


    
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() and args.gpu >=0 else "cpu")
    infer_device = device if args.infer_gpu else torch.device('cpu')
    # create  model
    if args.model == 'GraphSAGE':

        model = GraphSAGE(in_dim=node_in_dim,
                                        hidden_dim=args.node_hidden_dim, 
                                        num_class=num_classes, 
                                        num_layers=args.num_layers, 
                                        aggregator_type='mean',
                                        feat_drop=args.dropout,
                                        device=device)

    else:
        logging.info('The model \"{}\" is not implemented'.format(args.model))
        sys.exit(0)
    
    # send model to device
    model.to(device)

    # create optimizer

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn =  torch.nn.CrossEntropyLoss()


    # compute graph degre
    '''
    g.readonly()
    norm = 1. / g.in_degrees().float().unsqueeze(1)
    g.ndata['norm'] = norm
    # g.ndata['node_features'] = features

    degs = g.in_degrees().numpy()
    degs[degs > args.num_neighbors] = args.num_neighbors
    g.ndata['subg_norm'] = torch.FloatTensor(1./degs).unsqueeze(1)
    '''

    # create trainer
    checkpoint_path = os.path.join(log_path, str(args.model) + '_checkpoint.pt')


    pretrained_model_state_dict = {k:v for k, v in torch.load(args.pretrained_path).items()}

    model.load_pretrained(pretrained_model_state_dict)

    #for name, param in model.named_parameters():
    #    print(name, param.requires_grad)

    trainer = MiniBatchTrainer(  g=g, 
                                 model=model,
                                 loss_fn=loss_fn, 
                                 optimizer=optimizer, 
                                 epochs=args.epochs, 
                                 features=features, 
                                 labels=labels, 
                                 train_id=train_id, 
                                 val_id=val_id, 
                                 test_id=test_id,
                                 patience=args.patience, 
                                 batch_size=args.batch_size,
                                 test_batch_size=args.test_batch_size,
                                 num_neighbors=args.num_neighbors, 
                                 num_layers=args.num_layers,
                                 num_cpu=args.num_cpu, 
                                 device=device,
                                 infer_device=infer_device, 
                                 log_path=log_path,
                                 checkpoint_path=checkpoint_path)


    logging.info('Start training')
    start_time = datetime.datetime.now()
    best_val_result, test_result = trainer.train()
    end_time = datetime.datetime.now() 
    # recording the result
    
    line = [end_time.__str__()] + [args.model] + ['K=' + str(args.use_K)] + \
    [str(x) for x in best_val_result] + [str(x) for x in test_result] + \
    [str(args)] + [str((end_time-start_time).total_seconds())]
    line = ','.join(line) + '\n'

    with open(os.path.join(args.log_dir, str(args.model) + '_result.csv'), 'a') as f:
        f.write(line)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GTEA training')
    parser.add_argument("--data_dir", type=str, default='data/dynamic_eth',
            help="dataset name")
    parser.add_argument("--model", type=str, default='GraphSAGE',
            help="dataset name")    
    parser.add_argument("--use_K", type=int, default=None,
            help="select K-fold id, range from 0 to K-1")
    parser.add_argument("--K", type=int, default=5,
            help="Number of K in K-fold")
    parser.add_argument("--dropout", type=float, default=None,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--infer_gpu", action='store_false',
            help="infer device same as training device (default True)")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
            help="batch size")
    parser.add_argument("--max_event", type=int, default=20,
            help="max_event")
    parser.add_argument("--test_batch_size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num_neighbors", type=int, default=5,
            help="number of neighbors to be sampled")
    parser.add_argument("--node_hidden_dim", type=int, default=64,
            help="number of hidden gcn units")
    parser.add_argument("--time_hidden_dim", type=int, default=32,
            help="time layer dim")
    parser.add_argument("--num_layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--num_lstm_layers", type=int, default=1,
            help="number of hidden lstm layers")
    parser.add_argument("--num_heads", type=int, default=1,
            help="number of head for transformer")
    parser.add_argument("--bidirectional", type=bool, default=False,
            help="bidirectional lstm layer")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight_decay", type=float, default=0,
            help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=100,
            help="patience")
    parser.add_argument("--num_cpu", type=int, default=2,
            help="num_cpu")
    parser.add_argument("--log_dir", type=str, default='./experiment',
            help="experiment directory")
    parser.add_argument("--log_name", type=str, default='test',
            help="log directory name for this run")
    parser.add_argument("--remove_node_features", action='store_true')
    parser.add_argument("--pretrained_path", type=str, default=None, required=True,
            help="log directory name for this run")
    parser.add_argument("--train_ratio", type=float, default=0.6,
            help="ratio used to train new dataset")
    args = parser.parse_args()


    # print(args)

    main(args)
