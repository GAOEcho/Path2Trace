import json
import pickle
import random
from urllib.request import urlopen, quote

import networkx as nx
import numpy as np
import torch
from bidict import bidict
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
import sys

sys.path.append('../data/pengpai')
from address import address
from tqdm import tqdm

HEAD = 0
TAIL = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_neg_neb(neg_count, nebs, old2new_case_id):
    case_index = pickle.load(open('../data/pengpai/labeled_data/cases.pickle', 'rb'))
    case_pool = list(set(case_index).difference(set(nebs)))
    new_case_id_pool = [
        old2new_case_id[old_case_id] for old_case_id in case_pool
    ]
    if len(new_case_id_pool) <= neg_count:
        return list(new_case_id_pool)
    else:
        return random.sample(new_case_id_pool, neg_count)


def generate_pos_neg_samples(config, net, old2new_case_id):
    sample_type = config.sample_type

    proportion = config.proportion
    pos_idx_pair = []
    neg_idx_pair = []
    part_nodes = random.sample(set(net.nodes()), int(proportion * len(net.nodes())))

    for node in part_nodes:
        nebs = list(net.neighbors(node))
        pos_idx_pair.extend([node, neb, 1] for neb in nebs)
        neg_pair_count = int(round(random.uniform(1, config.lbd)))
        if neg_pair_count <= 0:
            neg_idx = get_neg_neb(neg_count=1, nebs=nebs, old2new_case_id=old2new_case_id)
        else:
            neg_idx = get_neg_neb(neg_pair_count, nebs, old2new_case_id)
        neg_idx_pair.extend([node, neg, 0] for neg in neg_idx)
    all_idx_pair = pos_idx_pair + neg_idx_pair
    with open(f'dataset/case_idx_pair/all_idx_pair_{str(config.lbd)}_{sample_type}.pickle', 'wb') as file:
        pickle.dump(file=file, obj=all_idx_pair)
        file.close()

    return all_idx_pair


def get_emb_label(padded_tensor_list, index_batch_x, device, case_confirm_date):
    y = []
    tensor_pairs = []
    i = 0
    d1, d2 = [], []
    length = len(padded_tensor_list[index_batch_x[0][i]])
    while (i < len(index_batch_x[0])):
        tensor_1 = padded_tensor_list[index_batch_x[0][i]]
        tensor_2 = padded_tensor_list[index_batch_x[1][i]]
        d1.append(case_confirm_date[index_batch_x[0][i].item()])
        d2.append(case_confirm_date[index_batch_x[1][i].item()])
        y.append(index_batch_x[2][i])
        tensor_pair = torch.cat((tensor_1, tensor_2))
        tensor_pairs.append(tensor_pair)
        i += 1

    path1 = torch.stack(tensor_pairs)[:, :length].to(device)
    path2 = torch.stack(tensor_pairs)[:, length:].to(device)
    d1 = torch.tensor(d1).to(device)
    d2 = torch.tensor(d2).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    return path1, path2, y, d1, d2


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file, protocol=2)
        file.close()


def get_index(tensor, tenser_list):
    for index, data in enumerate(tenser_list):
        if torch.equal(data, tensor):
            return index
    assert index != len(tenser_list) - 1, 'not found tensor'


def get_case_tensor_dict(case_paths, poi_emb):
    with open('../data/pengpai/labeled_data/nodeidxmaps.pickle', 'rb') as file:
        address_node2idx = pickle.load(file)
    poi2idxbi = bidict(address_node2idx['poi'])
    poi_emb_dic = {}
    if isinstance(poi_emb, torch.Tensor):
        for key, val in poi2idxbi.items():
            poi_emb_dic[key] = poi_emb[val]
        poi_emb = poi_emb_dic

    case_path_emb = {}
    for case_id, path in case_paths.items():
        case_path_emb[case_id] = []
        try:
            for addr in path:
                if addr.get_addr('poi') == '':
                    continue
                else:
                    case_path_emb[case_id].append(torch.Tensor(poi_emb[addr.get_addr('poi')]))
        except Exception as e:
            print('exception place: ', end='')
            print(e)
    case_tensor_dict = {}
    for case, path_emb in case_path_emb.items():
        try:
            case_tensor_dict[case] = torch.squeeze(torch.stack(path_emb).reshape((1, -1)))
        except Exception as e:
            case_tensor_dict[case] = torch.squeeze(torch.rand(1, 128))
    return case_tensor_dict


amap_ak = ''  # amap tooken


def amap_addr_query(address):
    url = 'https://restapi.amap.com/v3/geocode/geo?'
    query = quote(address)
    output = 'json'
    url2 = url + '&address=' + query + '&output=' + output + '&key=' + amap_ak
    req = urlopen(url2, timeout=10)
    res = req.read().decode()
    query_result = json.loads(res)
    return query_result


def load_file(file_path):
    with open(file_path, 'rb') as file:
        case_paths = pickle.load(file)
        file.close()
    return case_paths


def sort_reindex_case_tensor_dict(case_tensor_dict):
    case_tensor_dict_sorted_unpadded = sorted(case_tensor_dict.items(), key=lambda x: len(x[1]), reverse=True)
    i = 0
    old2new_case_id = bidict()
    reindexed_case_tensor_unpadded = []
    for old_case_ID, tensor in case_tensor_dict_sorted_unpadded:
        reindexed_case_tensor_unpadded.append(tensor)
        old2new_case_id[old_case_ID] = i
        i += 1
    return reindexed_case_tensor_unpadded, old2new_case_id


def reindex_sp_net(
        spread_net, old2new_case_id):
    reindexed_sp_net = nx.Graph()
    for node in spread_net.nodes():
        reindexed_sp_net.add_node(old2new_case_id[node])
    for edge in spread_net.edges():
        reindexed_sp_net.add_edge(old2new_case_id[edge[HEAD]], old2new_case_id[edge[TAIL]])
    return reindexed_sp_net


def get_all_loc():
    case_paths = load_file('../data/pengpai/labeled_data/case_path.pickle')
    poi_addr_bidict = {}
    for case_id, path in case_paths.items():
        for addr in path:
            poi_addr = addr.get_addr('poi')
            poi_addr_bidict[addr] = poi_addr
    addr2loc = {}
    for _, val in tqdm(poi_addr_bidict.items()):
        if val in addr2loc.keys():
            continue
        query_res = amap_addr_query(val)
        if (
                query_res['status'] == '1'
                and len(query_res['geocodes']) != 0
                and query_res['geocodes'][0]['location'] != ''
        ):
            addr2loc[val] = {}
            loc_str = query_res['geocodes'][0]['location']
            addr2loc[val]['log'] = float(loc_str.split(',')[0])
            addr2loc[val]['lat'] = float(loc_str.split(',')[1])
    with open('../data/pengpai/labeled_data/poi_addr_loc.pickle', 'wb') as file:
        pickle.dump(file=file, obj=addr2loc)
        file.close()
    return addr2loc


def find_optimal_cutoff_roc(TPR, FPR, Threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = Threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def ROC(label, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = find_optimal_cutoff_roc(TPR=tpr, FPR=fpr, Threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point


def draw_roc(y, y_hat):
    fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(y, y_hat)

    return roc_auc, fpr, tpr


def aggregate_importance(importances):
    return sum(importances) / len(importances)


def get_case_tensor(place_emb_method):
    case_paths = load_file('../data/pengpai/labeled_data/case_path.pickle')
    poi_embedding = load_file(
        f'../represent_learning/place_embeddings/{place_emb_method}.pickle'
    )
    assert len(case_paths) == 8604, f'missing {8604 - len(case_paths)} case'
    case_tensor_dict = get_case_tensor_dict(case_paths, poi_embedding)
    reindexed_case_tensor_unpadded, old2new_case_id = sort_reindex_case_tensor_dict(case_tensor_dict)
    assert (
            len(old2new_case_id.keys()) == 8604
    ), f'missing {8604 - len(old2new_case_id)} case'
    reindexed_case_tensor_padded = pad_sequence(reindexed_case_tensor_unpadded, batch_first=True)
    spread_net = load_file('spread_net.pickle')
    reindexed_sp_net = reindex_sp_net(spread_net, old2new_case_id)
    return reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded


def get_case_tensor_filter(poi_embedding):
    case_paths = load_file('../data/pengpai/labeled_data/case_path.pickle')
    assert len(case_paths) == 8604, f'missing {8604 - len(case_paths)} case'
    case_tensor_dict = get_case_tensor_dict(case_paths, poi_embedding)
    reindexed_case_tensor_unpadded, old2new_case_id = sort_reindex_case_tensor_dict(case_tensor_dict)
    assert (
            len(old2new_case_id.keys()) == 8604
    ), f'missing {8604 - len(old2new_case_id)} case'
    reindexed_case_tensor_padded = pad_sequence(reindexed_case_tensor_unpadded, batch_first=True)
    spread_net = load_file('spread_net.pickle')
    reindexed_sp_net = reindex_sp_net(spread_net, old2new_case_id)
    return reindexed_sp_net, old2new_case_id, reindexed_case_tensor_padded


def filter_place(dataloader, P2T_model, reindexed_case_tensor_padded):
    device = 'cpu'
    P2T_model.to(device)
    case_confirm_date = load_file('../data/pengpai/case_confirm_date.pickle')
    P2T_model.eval()
    count = 0
    for batch_x in dataloader:
        path1, path2, y_label, d1, d2 = get_emb_label(reindexed_case_tensor_padded, batch_x, device, case_confirm_date)
        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)
            y_hat_out, path1_emb, path2_emb, path1_weight, path2_weight = P2T_model(path1, path2, d1, d2)
            path1_weight = path1_weight.reshape([-1, 45, 45])
            path2_weight = path2_weight.reshape([-1, 45, 45])
            path1_mean = (path1_weight * torch.log(path1_weight)).sum(dim=2)
            path2_mean = (path2_weight * torch.log(path2_weight)).sum(dim=2)

            path_place_importance = torch.cat((path1_mean.reshape(-1), path2_mean.reshape(-1)))
            path = torch.cat((path1.reshape(128, 45, -1), path2.reshape(128, 45, -1)), 0)

            if count != 0:
                total_place_importance = torch.cat((total_place_importance, path_place_importance), 0)
                total_place_emb = torch.cat((total_place_emb, path), 0)
            else:
                total_place_emb = path
                total_place_importance = path_place_importance

            count += 1
    total_place_emb = total_place_emb.reshape([-1, 128]).cpu()
    total_place_importance = total_place_importance.reshape([-1, 1]).cpu()
    assert total_place_importance.shape[0] == total_place_emb.shape[0]
    total_place_emb, idx = total_place_emb.unique(dim=0, return_inverse=True)
    total_place_emb_importance = torch.cat((total_place_emb, torch.empty((total_place_emb.shape[0], 1))), 1)

    temp_dic = {}

    for i, item in enumerate(idx.tolist()):
        if item not in temp_dic.keys():
            temp_dic[item] = [total_place_importance[i].item()]
        else:
            temp_dic[item] += [total_place_importance[i].item()]

    for key, val in temp_dic.items():
        place_importance = aggregate_importance(val)
        total_place_emb_importance[key][-1] = place_importance

    assert total_place_emb_importance.shape[0] == len(temp_dic.keys())

    return total_place_emb, total_place_importance, total_place_emb_importance


def filter_tensor(total_place_emb_importance, path1, device, count, mask_ratio):
    temp_importance = torch.Tensor().to(device)
    for place in path1:
        try:
            temp_importance = torch.cat([temp_importance, total_place_emb_importance[
                torch.nonzero(torch.eq(total_place_emb_importance[:, :-1], place).all(dim=1)).item()][-1].unsqueeze(
                dim=0)])
        except ValueError:
            temp_importance = torch.cat([temp_importance, torch.Tensor([0]).to(device)])
            count += 1
    a = 0
    path1_emb_importance = torch.cat([path1, temp_importance.unsqueeze(dim=1)], dim=1)
    path1_emb_importance = path1_emb_importance.reshape([-1, 45, 129])

    path1 = path1.reshape([-1, 45, 128])
    val_place = torch.nonzero((path1 == 0).all(dim=-1) == False).tolist()
    val_len_dic = {}
    for i in range(len(val_place)):
        if val_place[i][0] not in val_len_dic.keys():
            val_len_dic[val_place[i][0]] = 1
        else:
            val_len_dic[val_place[i][0]] += 1
    for i in range(len(path1)):
        mask_c = val_len_dic[i]
        indice = torch.topk(path1_emb_importance[i, :mask_c, -1], max(1, round(mask_c * mask_ratio)), largest=False)[
            1].tolist()
        for j in indice:
            path1[i, j, :] = torch.zeros(128).to(device)

    return path1


def filter_all_place_tensor(total_place_emb_importance, mask_r):
    place_emb = load_file('../represent_learning/place_embeddings/hprl_carl.pickle')

    temp_importance = total_place_emb_importance[:, -1].tolist()

    for place, emb in place_emb.items():
        try:
            imp = temp_importance[
                torch.nonzero(torch.eq(total_place_emb_importance[:, :-1], torch.tensor(emb)).all(dim=1)).item()]
        except ValueError:
            imp = -10

        place_emb[place] = np.concatenate((emb, [imp]))

    keys = list(place_emb.keys())
    embs = torch.tensor(list(place_emb.values()))

    filter_indices = torch.topk(embs[:, -1], k=int(len(temp_importance) * mask_r), largest=False).indices

    embs[filter_indices] = torch.zeros((filter_indices.shape[0], embs.shape[1]), dtype=embs.dtype)

    res = {}
    for i in range(len(keys)):
        res[keys[i]] = np.array(embs[i][:-1])
    return res


def filter_all_place_tensor_random(total_place_emb_importance, mask_r):
    place_emb = load_file('../represent_learning/place_embeddings/hprl_carl.pickle')

    temp_importance = total_place_emb_importance[:, -1].tolist()

    keys = list(place_emb.keys())
    embs = torch.tensor(list(place_emb.values()))

    filter_indices = torch.tensor(random.sample(range(embs.shape[0]), int(len(temp_importance) * mask_r)))

    embs[filter_indices] = torch.zeros((filter_indices.shape[0], embs.shape[1]), dtype=embs.dtype)

    res = {}
    for i in range(len(keys)):
        res[keys[i]] = np.array(embs[i])
    return res


def filter_tensor_random(path1, device, mask_ratio):
    path1 = path1.reshape([-1, 45, 128])
    val_place = torch.nonzero((path1 == 0).all(dim=-1) == False).tolist()
    val_len_dic = {}
    for i in range(len(val_place)):
        if val_place[i][0] not in val_len_dic.keys():
            val_len_dic[val_place[i][0]] = 1
        else:
            val_len_dic[val_place[i][0]] += 1
    for i in range(len(path1)):
        mask_c = val_len_dic[i]
        mask = random.sample(range(0, mask_c), max(1, round(mask_c * mask_ratio)))
        for m in mask:
            path1[i, m, :] = torch.zeros(128).to(device)
    return path1


def test_with_filter(dataloader, P2T_model, reindexed_case_tensor_padded, config):
    device = torch.device('cuda:{}'.format(config.gpu)) if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    case_confirm_date = load_file('../data/pengpai/case_confirm_date.pickle')
    roc, prt = {}, {}
    y, y_hat = [], []

    P2T_model.eval()
    for batch_x in dataloader:
        path1, path2, y_label, d1, d2 = get_emb_label(reindexed_case_tensor_padded, batch_x, device, case_confirm_date)

        path1 = path1.reshape([-1, 128])
        path2 = path2.reshape([-1, 128])

        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)

            y_hat_out, path1_emb, path2_emb, _, _ = P2T_model(path1, path2, d1, d2)
            y_hat_out = y_hat_out.squeeze()
            y_hat += y_hat_out.cpu()
            y += y_label.cpu()
    roc['fpr'], roc['tpr'], roc['thresholds'] = metrics.roc_curve(y, y_hat, pos_label=1)
    prt['precision'], prt['recall'], prt['thresholds'] = metrics.precision_recall_curve(y, y_hat)
    auc = metrics.auc(roc['fpr'], roc['tpr'])

    return auc


def test_with_filter_random(dataloader, P2T_model, reindexed_case_tensor_padded):
    device = 'cpu'
    case_confirm_date = load_file('../data/pengpai/case_confirm_date.pickle')

    roc, prt = {}, {}
    y, y_hat = [], []

    P2T_model.eval()

    for batch_x in dataloader:
        path1, path2, y_label, d1, d2 = get_emb_label(reindexed_case_tensor_padded, batch_x, device, case_confirm_date)

        path1 = path1.reshape([-1, 128])
        path2 = path2.reshape([-1, 128])

        with torch.no_grad():
            path1 = path1.reshape([128, 45, 128]).permute(1, 0, 2)
            path2 = path2.reshape([128, 45, 128]).permute(1, 0, 2)

            y_hat_out, path1_emb, path2_emb, _, _ = P2T_model(path1, path2, d1, d2)
            y_hat_out = y_hat_out.squeeze()
            y_hat += y_hat_out.cpu()
            y += y_label.cpu()
    roc['fpr'], roc['tpr'], roc['thresholds'] = metrics.roc_curve(y, y_hat, pos_label=1)
    prt['precision'], prt['recall'], prt['thresholds'] = metrics.precision_recall_curve(y, y_hat)
    auc = metrics.auc(roc['fpr'], roc['tpr'])

    return auc
