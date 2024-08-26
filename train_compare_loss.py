"""
Compare the loss between the Knowledge NN and the Vanilla NN under full dataset.
"""
import argparse
import os.path
import h5py
import numpy as np
import torch
from torch import nn
from algo.pinn import NN, PINN
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import Feeder, set_random_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--ratios', type=list, default=[0.01, 0.1, 0.3, 0.5, 1])
    parser.add_argument('--model_types', type=list, default=["VanillaNN", "KnowledgeNN"])
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--hidden_size', type=tuple, default=(128, 128))
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--idm_data_path', type=str, default='./data/idm_data.h5')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint/compare_loss/')
    parser.add_argument('--logs_path', type=str, default='./logs/compare_loss/')

    args = parser.parse_args()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    for model_type in args.model_types:
        writer = SummaryWriter(args.logs_path + '{}_{}'.format(model_type,
                                                               datetime.datetime.now().strftime(
                                                                   "%Y-%m-%d_%H-%M-%S")))
        set_random_seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(">>>>>>>>>>>> Training on ", device)

        # idm_data contains : 1. relative position;        2. relative velocity
        #                     3. velocity of the follower; 4. acceleration of the follower
        with h5py.File(args.idm_data_path, 'r') as hf:
            datasets = list(hf.keys())
            idm_data = [hf[dataset][:] for dataset in datasets]

        idm_data = np.vstack(idm_data)  # idm_data [262630, 4] [dx, dv, vf, a]

        train_feeder = Feeder(idm_data, train_val_test='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_feeder, batch_size=args.batch_size,
                                                   shuffle=True)
        val_feeder = Feeder(idm_data, train_val_test='val')
        val_loader = torch.utils.data.DataLoader(dataset=val_feeder, batch_size=args.batch_size, shuffle=True)

        if model_type.lower().find("vanilla") != -1:
            model = NN(args.input_dim, args.output_dim, hidden_size=args.hidden_size,
                       activation=args.activation, device=device)
        else:
            model = PINN(args.input_dim, args.output_dim, hidden_size=args.hidden_size,
                         activation=args.activation, device=device)

        # train and eval model
        best_mse = 100000
        best_it = 0

        train_total_loss = []
        val_total_loss = []
        loss = nn.MSELoss()
        best_val_loss = np.float32('inf')
        best_epoch = 0
        for epoch in tqdm(range(args.epochs)):
            running_loss = 0
            model.train()
            for data in train_loader:
                train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_label = train_label.to(device)
                prediction = model(train_feature)
                train_loss = loss(train_label, prediction)
                running_loss += train_loss.item()
                model.optimizer.zero_grad()
                train_loss.backward()
                model.optimizer.step()

            train_total_loss.append(running_loss)
            writer.add_scalar('train_loss_', running_loss, epoch + 1)
            model.eval()
            validaton_loss = 0
            for data in val_loader:
                val_feature, val_label = data
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)
                prediction = model(val_feature)
                val_loss = loss(val_label, prediction)
                validaton_loss += val_loss.item()

            writer.add_scalar('val_loss', validaton_loss, epoch + 1)
            val_total_loss.append(validaton_loss)
            if validaton_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = validaton_loss
                torch.save(model.state_dict(), args.model_save_path + '{}_best_val.pth'.format(model_type))

        if model_type.lower().find("vanilla") != -1:
            best_model = NN(args.input_dim, args.output_dim, hidden_size=args.hidden_size,
                            activation=args.activation)
        else:
            best_model = PINN(args.input_dim, args.output_dim, hidden_size=args.hidden_size,
                              activation=args.activation)

        best_model.load_state_dict(torch.load(args.model_save_path + '{}_best_val.pth'.format(model_type)))

        # testing set
        test_feeder = Feeder(idm_data, train_val_test='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_feeder, batch_size=1, shuffle=False)

        test_total_SE = []
        for data in test_loader:
            test_feature, test_label = data
            prediction = best_model(test_feature)
            test_SE = (test_label - prediction) ** 2
            test_total_SE.append(test_SE.item())
        test_RMSE_loss = np.sqrt(np.mean(test_total_SE))
        print("{} | best epoch {} | RMSE is {}".format(model_type, best_epoch, test_RMSE_loss))

        writer.close()
