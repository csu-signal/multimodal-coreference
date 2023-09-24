from helper import *
import pickle
import numpy as np
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
import torch
from tqdm import tqdm
from helper import cluster
import pandas as pd
import time

def read(key, response):
    return get_coref_infos('%s' % key, '%s' % response,
            False, False, True)

def fetch_cluster_scores(dataset_folder, evt_mention_map, all_mention_pairs, dataset, split, heu, similarities, dpos_score_map, out_name, threshold):
    curr_mentions = sorted(evt_mention_map.keys())

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, evt_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = dataset_folder + f'/evt_gold_{split}.keyfile'
    generate_key_file(curr_gold_cluster_map, 'evt', dataset_folder, gold_key_file)
    score_map  = list(dpos_score_map.values())

    mid2cluster = cluster(curr_mentions, all_mention_pairs, score_map, threshold)
    system_key_file = dataset_folder + f'/evt_gold_dpos_{out_name}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', dataset_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)

    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    conf = np.round((mf + bf + cf) / 3, 1)
    print(dataset, split)
    result_string = f'& {heu} && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)
    return conf


def predict_with_dpos(dataset, split, dpos_score_map, heu='lh_oracle', threshold=0.5):
    dataset_folder = f'./datasets/{dataset}/'
    #threshold =0.93 #for the random initialization
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    _, _, _, fns = mps_trans
    tps, fps, tns, fns_nt = mps
    print(len(tps), len(fps), len(fns))
    all_mention_pairs = tps + fps
    heu_predictions = np.array([1] * len(tps) + [0] * len(fps)) #ignore this
    fetch_cluster_scores(dataset_folder, evt_mention_map, all_mention_pairs, dataset, split, heu, heu_predictions, dpos_score_map, out_name=heu, threshold=threshold)

def generate_ensemble_results(image_type = "generated"):
    dataset = 'ecb'
    split = 'test'
    heu = 'lh_oracle'
    dataset_folder = f'./datasets/{dataset}'

    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    #split ='train'
    print("split", split)
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}
    curr_mentions = list(evt_mention_map.keys())
    # dev_pairs, dev_labels = zip(*load_lemma_dataset('./datasets/ecb/lemma_balanced_tp_fp_test.tsv'))
    #split ='train'
    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    tps, fps, tns, fns = mps

    tps = tps
    fps = fps

    test_pairs = tps + fps
    test_labels = [1]*len(tps) + [0]*len(fps)
    print(len(test_pairs), len(evt_mention_map))
    df_real['new_pairs'] = test_pairs
    if image_type == "generated":
        ensemble_feature_list  = ['VIT_LM','LM_VIT', 'LM_SWIN', 'SWIN_LM', 'BEIT_LM', 'LM_BEIT', 'LM_CLIP',
               'CLIP_LM', 'VIT+LM', 'SWIN+LM', 'BEIT+LM', 'CLIP+LM']
    else:
        ensemble_feature_list = ['VIT_LM_real', 'LM_VIT_real', 'SWIN_LM_real', 'LM_SWIN_real',
       'BEIT_LM_real', 'LM_BEIT_real', 'CLIP_LM_real', 'LM_CLIP_real']
    df_hard_pos = df_real.loc[(df_real['coref_label']==1) & (df_real['sem_label']==0) ] #hard examples pos
    df_hard_neg = df_real.loc[(df_real['coref_label']==0) & (df_real['sem_label']==0) ] #hard examples neg
    df_easy_pos = df_real.loc[(df_real['coref_label']==1) & (df_real['sem_label']==1) ] # easy pos
    df_easy_neg = df_real.loc[(df_real['coref_label']==0) & (df_real['sem_label']==1) ] # easy easy neg
    e_p_list = df_easy_pos['new_pairs'].to_list()
    e_h_list = df_easy_neg['new_pairs'].to_list()
    h_p_list = df_hard_pos['new_pairs'].to_list()
    h_n_list = df_hard_neg['new_pairs'].to_list()

    for x, y in enumerate(ensemble_feature_list):
        print(y)
        easy_dict_pos = {(k[0], k[1]):v for k, v in zip(e_p_list, list(df_easy_pos['LM_prediction']))}
        easy_dict_neg = {(k[0], k[1]):v for k, v in zip(e_h_list, list(df_easy_neg['LM_prediction']))}
        hard_dict_pos = {(k[0], k[1]):v for k, v in zip(h_p_list, list(df_hard_pos['LM_BEIT']))} # generated images are more likely to resolve harder positively coreferring pairs
        hard_dict_neg = {(k[0], k[1]):v for k, v in zip(h_n_list, list(df_hard_neg[y]))}
        test_pairs_dict = {pair:value for pair,value in zip(test_pairs, test_labels)} #get the actual map of inference pair-labels
        test_pairs_dict = {key: easy_dict_pos.get(key, test_pairs_dict[key]) for key in test_pairs_dict}
        test_pairs_dict = {key: easy_dict_neg.get(key, test_pairs_dict[key]) for key in test_pairs_dict}
        test_pairs_dict = {key: hard_dict_pos.get(key, test_pairs_dict[key]) for key in test_pairs_dict}
        test_pairs_dict = {key: hard_dict_neg.get(key, test_pairs_dict[key]) for key in test_pairs_dict}
        shared_items = {k: test_pairs_dict_original[k] for k in test_pairs_dict_original if k in test_pairs_dict and test_pairs_dict_original[k] == test_pairs_dict[k]}
        print(f"Shared final scores for ensemble of LLM with {y}:{len(shared_items)}")
        print(f"Final Emsemble CDCR Scores for {y} with {image_type} images")
        predict_with_dpos(dataset, split, dpos_score_map=test_pairs_dict, heu='lh_oracle', threshold=0.5)

if __name__ == '__main__':
    generate_ensemble_results(image_type = "generated")