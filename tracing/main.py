import argparse
import os

import torch.utils.data
from dataloader.dataloader import get_dataloader, get_disjoint_dataloader
from model.loc_model import loc_model
from model.P2T import P2T
import torch.nn as nn
from utils import *
import sys

sys.path.append('../data/pengpai')
from address import address


def train(train_dataloader, config, reindexed_case_tensor_padded):
    batch_size = config.batch_size
    path_len = config.path_len
    place_emb_dim = int(reindexed_case_tensor_padded.shape[1] / path_len)
    device = torch.device(
        f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    case_confirm_date = load_file('../data/pengpai/case_confirm_date.pickle')
    P2T_model = P2T(place_emb_dim=place_emb_dim, path_len=path_len)
    if config.rl_type == 'loc':
        P2T_model = loc_model()
    opt_P2T = torch.optim.Adam(P2T_model.parameters(), lr=config.lr_P2T)
    P2T_loss = nn.MSELoss()

    P2T_model.to(device)

    epochs = tqdm(range(config.num_epochs), ncols=100)
    count = 0
    for _ in epochs:
        for batch_x in train_dataloader:
            path1, path2, y, d1, d2 = get_emb_label(reindexed_case_tensor_padded, batch_x, device, case_confirm_date)

            opt_P2T.zero_grad()
            path1 = path1.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            path2 = path2.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            y_hat, path1_emb, path2_emb, _, _ = P2T_model(path1, path2, d1, d2)
            y = y.float()
            y_hat = y_hat.squeeze()
            P2T_l = P2T_loss(y, y_hat)
            P2T_l.backward()
            opt_P2T.step()
            count += 1

        epochs.set_postfix(P2T_loss=P2T_l.cpu().item())

    return P2T_model


def test(test_dataloader, config, P2T_model, reindexed_case_tensor_padded):
    y, y_hat = [], []
    batch_size = config.batch_size
    path_len = config.path_len
    place_emb_dim = int(reindexed_case_tensor_padded.shape[1] / path_len)
    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else 'cpu')
    case_confirm_date = load_file('../data/pengpai/case_confirm_date.pickle')
    P2T_model.to(device)
    for batch_x in test_dataloader:
        path1, path2, y_label, d1, d2 = get_emb_label(reindexed_case_tensor_padded, batch_x, device, case_confirm_date)
        with torch.no_grad():
            path1 = path1.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            path2 = path2.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            y_hat_out, _, _, _, _ = P2T_model(path1, path2, d1, d2)
            y_hat_out = y_hat_out.squeeze()
            y_label = y_label.float()
        y_hat += y_hat_out.cpu()
        y += y_label.cpu()

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=1)
    opt_thd, opt_point = find_optimal_cutoff_roc(tpr, fpr, thresholds)
    y_b_hat = y_hat.copy()
    for i, val in enumerate(y_hat):
        if val > opt_thd:
            y_b_hat[i] = 1
        else:
            y_b_hat[i] = 0
    acc = metrics.accuracy_score(y, y_b_hat)
    precision = metrics.precision_score(y, y_b_hat)
    recall = metrics.recall_score(y, y_b_hat)
    f1 = metrics.f1_score(y, y_b_hat)
    auc = metrics.roc_auc_score(y, y_hat)

    print(f'accurancy {acc} \t precision {precision} \t recall {recall} \t f1 {f1} \t auc {auc}')

    return [acc, precision, recall, f1, auc]


def reconstruct_spread_graph(test_dataloader, config, P2T_model, reindexed_case_tensor_padded):
    y, y_hat = [], []
    batch_size = config.batch_size
    path_len = config.path_len
    place_emb_dim = int(reindexed_case_tensor_padded.shape[1] / path_len)
    device = torch.device(
        f"cuda:{config.gpu}" if torch.cuda.is_available() else 'cpu'
    )
    P2T_model.to(device)
    edge_list_h = []
    edge_list_t = []
    edge_list_y = []
    for batch_x in test_dataloader:
        path1, path2, y_label = get_emb_label(reindexed_case_tensor_padded, batch_x, device)
        with torch.no_grad():
            path1 = path1.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            path2 = path2.reshape([batch_size, path_len, place_emb_dim]).permute(1, 0, 2)
            y_hat_out, _, _, _, _ = P2T_model(path1, path2)
            y_hat_out = y_hat_out.squeeze()
            y_label = y_label.float()
        y_hat += y_hat_out.cpu()
        y += y_label.cpu()
        edge_list_h += batch_x[0].tolist()
        edge_list_t += batch_x[1].tolist()
        edge_list_y += batch_x[2].tolist()

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=1)
    opt_thd, opt_point = find_optimal_cutoff_roc(tpr, fpr, thresholds)
    y_b_hat = y_hat.copy()
    for i, val in enumerate(y_hat):
        if val > opt_thd:
            y_b_hat[i] = 1
        else:
            y_b_hat[i] = 0
    edge_list_y_b_hat = y_b_hat
    acc = metrics.accuracy_score(y, y_b_hat)
    precision = metrics.precision_score(y, y_b_hat)
    recall = metrics.recall_score(y, y_b_hat)
    f1 = metrics.f1_score(y, y_b_hat)
    auc = metrics.roc_auc_score(y, y_hat)

    print(f'accuracy {acc} \t precision {precision} \t recall {recall} \t f1 {f1} \t auc {auc}')

    if not os.path.exists('saved/results/reconstruct_graph/'):
        os.mkdir('saved/results/reconstruct_graph/')
    else:
        with open('saved/results/reconstruct_graph/reconstruct_edge_list.pickle', 'wb') as file:
            pickle.dump(obj=np.array([edge_list_h, edge_list_t, edge_list_y, edge_list_y_b_hat]), file=file)


def get_data(config):
    reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded = get_case_tensor(config.rl_type)
    print(f'train with lambda:{config.lbd},train ratio:{config.train_ratio},lr:{config.lr_P2T}')
    train_dataloader, test_dataloader, all_dataloader = get_dataloader(config, reindexed_sp_net, old2new_case_id)

    return reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader


def get_disjoint_data(config):
    reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded = get_case_tensor(config.rl_type)
    print(f'train with lambda:{config.lbd},train ratio:{config.train_ratio},lr:{config.lr_P2T}')
    train_dataloader, test_dataloader, all_dataloader = get_disjoint_dataloader(config, reindexed_sp_net,
                                                                                old2new_case_id)
    return reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader


def train_on_model(model_name_list):
    test_res = {}
    for rl_type in model_name_list:
        print('train on {} ...'.format(rl_type))
        config.rl_type = rl_type
        reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader = get_data(
            config)
        P2T_model = train(train_dataloader, config, reindexed_case_tensor_padded)
        test_res[rl_type] = test(test_dataloader, config, P2T_model, reindexed_case_tensor_padded)
    return test_res


def train_test_seed(model_name_list):
    res_seeds = load_file('res_seed.pickle')
    seeds = res_seeds.keys()
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        print(f'train {i + 1}/{len(seeds)} on seed {seed} ...')
        res_seeds[seed] = train_on_model(model_name_list)

        save_pickle(obj=res_seeds, filepath='saved/res_seed.pickle')
    return res_seeds


def main(config):
    setup_seed(20)
    if config.run_model == 'baselines':
        baselines = ['spacene', 'louvainne', 'deepwalk', 'node2vec', 'randne', 'boostne', 'sdne', 'gae', 'vgae']
        train_on_model(baselines)

    if config.run_model == 'ablation':
        baselines = ['hprl', 'loc', 'carl_loc']
        train_on_model(baselines)

    if config.run_model == 'train':
        reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader = get_data(
            config)
        P2T_model = train(train_dataloader, config, reindexed_case_tensor_padded)
        _ = test(test_dataloader, config, P2T_model, reindexed_case_tensor_padded)

    if config.run_model == "minority_sample_ana":

        res = {}
        if os.path.exists('saved/minority_ana_with_seed_ratio.pickle'):
            res = pickle.load(open('saved/minority_ana_with_seed_ratio.pickle', 'rb'))
            seeds = res.keys()
        else:
            seeds = [random.randint(0, 100000) for _ in range(5)]
        print(f'seeds: {seeds}')
        for seed in seeds:
            if seed not in res.keys():
                res[seed] = {}
            setup_seed(seed)
            print(f'training on seed: {seed}......')
            for i in range(1, 10):
                config.train_ratio = i / 10.0
                config.num_epochs = 80
                if config.train_ratio not in res[seed].keys():
                    reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader = get_data(config)
                    P2T_model = train(train_dataloader, config, reindexed_case_tensor_padded)
                    res[seed][config.train_ratio] = test(test_dataloader, config, P2T_model,
                                                         reindexed_case_tensor_padded)

                    pickle.dump(res, open('saved/minority_ana_with_seed_ratio.pickle', 'wb'))

    if config.run_model == "disjoint_train":
        print('train with disjoint data')
        res = {}
        if os.path.exists('saved/disjoint_ana_with_seed_ratio.pickle'):
            res = pickle.load(open('saved/disjoint_ana_with_seed_ratio.pickle', 'rb'))
            seeds = res.keys()
        else:
            seeds = [random.randint(0, 100000) for _ in range(5)]
        print(f'seeds: {seeds}')
        for seed in seeds:
            if seed not in res.keys():
                res[seed] = {}
            setup_seed(seed)
            print(f'training on seed: {seed}......')
            for i in range(1, 10):
                config.train_ratio = i / 10.0
                config.num_epochs = 80
                if config.train_ratio not in res[seed].keys():
                    reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader = get_disjoint_data(
                        config)
                    P2T_model = train(train_dataloader, config, reindexed_case_tensor_padded)
                    res[seed][config.train_ratio] = test(test_dataloader, config, P2T_model,
                                                         reindexed_case_tensor_padded)
                    pickle.dump(res, open('saved/disjoint_ana_with_seed_ratio.pickle', 'wb'))

    if config.run_model == "get_avg":
        model_list = ['hprl_carl', 'carl_loc', 'hprl', 'loc', 'spacene', 'louvainne', 'deepwalk', 'node2vec',
                      'nodesketch', 'randne', 'boostne', 'sdne', 'gae', 'vgae']

        train_test_seed(model_list)

    if config.run_model == 'graph_recon':
        reindexed_case_tensor_padded, train_dataloader, test_dataloader, all_dataloader = get_data(
            config)
        P2T_model = train(train_dataloader, config, reindexed_case_tensor_padded)
        reconstruct_spread_graph(all_dataloader, config, P2T_model, reindexed_case_tensor_padded)

    if config.run_model == 'lr_lbd_sen':
        config.lr_P2T = 1e-5
        lr = config.lr_P2T
        res = load_file('saved/sen_res.pickle')
        for i in range(0, 10):
            config.lr_P2T = lr * (2 ** i)
            if i not in res.keys():
                res[i] = {}
            for j in range(1, 20):
                if j in res[i].keys():
                    continue
                print('train in epoch:{}'.format(i))
                print('learning rate: {}'.format(config.lr_P2T))
                config.lbd = j
                print('lambda={}'.format(config.lbd))
                place_train_dataloader, reindexed_case_tensor_padded, place_test_dataloader, train_dataloader, \
                test_dataloader, all_dataloader = get_data(config)

                P2T_model = train(place_train_dataloader, config, reindexed_case_tensor_padded)
                P2T_auc = test(place_test_dataloader, config, P2T_model, reindexed_case_tensor_padded)
                res[i][j] = P2T_auc
                save_pickle(res, 'saved/sen_res.pickle')

        save_pickle(res, 'saved/sen_res.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_P2T', type=float, default=18e-5, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=128, help="P2T batch size")
    parser.add_argument("--sample_type", type=str, default="random", help="P2T batch size")
    parser.add_argument("--model_save_dir", type=str, default="saved/model", help="save dir for model")
    parser.add_argument("--HEAD", type=int, default=0, help="")
    parser.add_argument("--TAIL", type=int, default=1, help="")
    parser.add_argument("--LABEL", type=int, default=2, help="")
    parser.add_argument("--gpu", type=int, default=2, help="gpu device")
    parser.add_argument("--proportion", type=float, default=1.0, help="proportion of the total data for train and test")
    parser.add_argument("--run_model", type=str, default='train', help="run options")
    parser.add_argument("--lbd", type=int, default=4, help="hyperparameter for sample positive and negative samples")
    parser.add_argument('--path_len', type=int, default=45, help="max Path length")
    parser.add_argument('--rl_type', type=str, default='hprl_carl', help="representation learning method name")
    parser.add_argument('--res_save_path', type=str, default='saved/results', help="results save Path")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="train ratio of dataset")
    config = parser.parse_args(args=[])
    main(config)
