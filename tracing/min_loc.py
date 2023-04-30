import argparse
from torch.utils.tensorboard import SummaryWriter
from dataloader.dataloader import get_dataloader
from utils import *
import torch.utils.data
from model.P2T import *
import sys

sys.path.append('../data/pengpai')
from address import address


def train(train_dataloader, config, reindexed_case_tensor_padded):
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    P2T_model = P2T(128, config.path_len)
    opt_P2T = torch.optim.Adam(P2T_model.parameters(), lr=config.lr_P2T)
    case_confirm_date = load_file('../data/pengpai/case_confirm_date.pickle')
    P2T_loss = nn.MSELoss()
    P2T_model.to(device)

    epochs = tqdm(range(config.num_epochs), ncols=100)
    count = 0
    for _ in epochs:
        for batch_x in train_dataloader:
            path1, path2, y, d1, d2 = get_emb_label(reindexed_case_tensor_padded, batch_x, device, case_confirm_date)

            opt_P2T.zero_grad()
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)
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
    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else 'cpu')
    case_confirm_date = load_file('../data/pengpai/case_confirm_date.pickle')

    P2T_model.to(device)

    for batch_x in test_dataloader:
        path1, path2, y_label, d1, d2 = get_emb_label(reindexed_case_tensor_padded, batch_x, device, case_confirm_date)
        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)
            y_hat_out, _, _, _, _ = P2T_model(path1, path2, d1, d2)
            y_hat_out = y_hat_out.squeeze()
            y_label = y_label.float()
        y_hat += y_hat_out.cpu()
        y += y_label.cpu()

    auc, fpr, tpr = draw_roc(y, y_hat)

    return auc


def get_data(config):
    reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded = get_case_tensor('hprl_carl')
    train_dataloader, test_dataloader, all_dataloader = get_dataloader(config, reindexed_sp_net, old2new_case_id)

    place_train_dataset, place_test_dataset = torch.utils.data.random_split(train_dataloader.dataset,
                                                                            [len(train_dataloader.dataset) // 2,
                                                                             len(train_dataloader.dataset) - len(
                                                                                 train_dataloader.dataset) // 2])

    place_train_dataloader = torch.utils.data.DataLoader(place_train_dataset, batch_size=config.batch_size,
                                                         shuffle=True, drop_last=True)
    place_test_dataloader = torch.utils.data.DataLoader(place_test_dataset, batch_size=config.batch_size, shuffle=True,
                                                        drop_last=True)

    return place_train_dataloader, reindexed_case_tensor_padded, place_test_dataloader, train_dataloader, test_dataloader, all_dataloader


def get_data_from_place_emb(config, filter_place_emb):
    reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded = get_case_tensor_filter(filter_place_emb)
    train_dataloader, test_dataloader, all_dataloader = get_dataloader(config, reindexed_sp_net, old2new_case_id)
    place_train_dataset, place_test_dataset = torch.utils.data.random_split(train_dataloader.dataset,
                                                                            [len(train_dataloader.dataset) // 2,
                                                                             len(train_dataloader.dataset) - len(
                                                                                 train_dataloader.dataset) // 2])

    place_train_dataloader = torch.utils.data.DataLoader(place_train_dataset, batch_size=config.batch_size,
                                                         shuffle=True, drop_last=True)
    place_test_dataloader = torch.utils.data.DataLoader(place_test_dataset, batch_size=config.batch_size, shuffle=True,
                                                        drop_last=True)

    return place_train_dataloader, reindexed_case_tensor_padded, place_test_dataloader, train_dataloader, test_dataloader, all_dataloader


def filter_data(config, total_place_emb_importance, mask_r):
    filter_place_emb = filter_all_place_tensor(total_place_emb_importance, mask_r)
    return get_data_from_place_emb(config, filter_place_emb)


def filter_data_random(config, total_place_emb_importance, mask_r):
    filter_place_emb = filter_all_place_tensor_random(total_place_emb_importance, mask_r)
    return get_data_from_place_emb(config, filter_place_emb)


def min_loc(config):
    _, reindexed_case_tensor_padded, _, train_dataloader, test_dataloader, all_dataloader = get_data(
        config)
    P2T_model = train(train_dataloader, config, reindexed_case_tensor_padded)
    P2T_auc = test(test_dataloader, config, P2T_model, reindexed_case_tensor_padded)
    print(f'P2T auc {P2T_auc}')
    total_place_emb, total_place_importance, total_place_emb_importance = filter_place(all_dataloader, P2T_model,
                                                                                       reindexed_case_tensor_padded)

    base = 100.0
    with open('saved/min_loc.log', 'w') as file:
        print('mask ratio, min auc, random auc')
        for topk in range(1, 100):
            _, reindexed_case_tensor_padded, _, train_dataloader, test_dataloader, all_dataloader = filter_data(config,
                                                                                                                total_place_emb_importance,
                                                                                                                topk / base)
            min_auc = test_with_filter(test_dataloader, P2T_model, reindexed_case_tensor_padded, config)

            _, reindexed_case_tensor_padded, _, train_dataloader, test_dataloader, all_dataloader = filter_data_random(
                config,
                total_place_emb_importance,
                topk / base)
            random_auc = test_with_filter_random(test_dataloader, P2T_model, reindexed_case_tensor_padded)

            print(f'{topk / base}, {min_auc}, {random_auc}')


def main(config):
    setup_seed(10)
    min_loc(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='att_P2T', help='model name ')
    parser.add_argument('--lr_P2T', type=float, default=18e-5, help='learning rate')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of neg samples and pos samples')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=128, help="P2T batch size")
    parser.add_argument("--model_save_dir", type=str, default="saved/model/", help="save dir for model")
    parser.add_argument("--HEAD", type=int, default=0, help="")
    parser.add_argument("--lbd", type=int, default=4)
    parser.add_argument('--path_len', type=int, default=45, help="max Path length")
    parser.add_argument("--TAIL", type=int, default=1, help="")
    parser.add_argument("--LABEL", type=int, default=2, help="")
    parser.add_argument("--sample_type", type=str, default='random',
                        help="sample type for pos and neg case pairs")
    parser.add_argument("--gpu", type=int, default=4, help="gpu device")
    parser.add_argument("--proportion", type=float, default=1.0, help="proportion of the total data for train and test")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    config = parser.parse_args(args=[])
    main(config)
