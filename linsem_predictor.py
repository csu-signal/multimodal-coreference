import sklearn.linear_model
import pickle
import numpy as np
from helper import *
import pickle
import numpy as np
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
import torch
from models import Lin_Sem_Predictor
from tqdm import tqdm
from heuristic import lh_split
from helper import cluster
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import pandas as pd
import time

def find_lh_lh_oracle_overlap(dataset, dataset_folder):
    lh_dict_train  = {}
    lh_oracle_dict_train  = {}
    lh_dict_test  = {}
    lh_oracle_dict_test  = {}

    common_ids = {}
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))

    # LH Oracle
    train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_oracle/mp_mp_t_train.pkl', 'rb'))
    dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_oracle/mp_mp_t_dev.pkl', 'rb'))
    test_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_oracle/mp_mp_t_test.pkl', 'rb'))
    tps_train, fps_train, _, _ = train_mp_mpt
    tps_dev, fps_dev, _, _ = dev_mp_mpt
    tps_test, fps_test, _, _ = test_mp_mpt
    train_pairs_lh_oracle = list(tps_train + fps_train)
    train_labels_lh_oracle = [1]*len(tps_train) + [0]*len(fps_train)
    dev_pairs_lh_oracle = list(tps_dev + fps_dev)
    dev_labels_lh_oracle = [1] * len(tps_dev) + [0] * len(fps_dev)
    test_pairs_lh_oracle = list(tps_test + fps_test)
    test_labels_lh_oracle = [1] * len(tps_test) + [0] * len(fps_test)

    # LH
    train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_train.pkl', 'rb'))
    dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_dev.pkl', 'rb'))

    test_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_test.pkl', 'rb'))

    tps_train, fps_train, _, _ = train_mp_mpt
    tps_dev, fps_dev, _, _ = dev_mp_mpt
    tps_test, fps_test, _, _ = test_mp_mpt


    train_pairs_lh = list(tps_train + fps_train)
    train_labels_lh = [1]*len(tps_train) + [0]*len(fps_train)
    dev_pairs_lh = list(tps_dev + fps_dev)
    dev_labels_lh = [1] * len(tps_dev) + [0] * len(fps_dev)
    test_pairs_lh = list(tps_test + fps_test)
    test_labels_lh = [1] * len(tps_test) + [0] * len(fps_test)

    for x, y in zip(train_pairs_lh,train_labels_lh) :
        lh_dict_train[x] = y

    for x, y in zip(train_pairs_lh_oracle,train_labels_lh_oracle) :
        lh_oracle_dict_train[x] = y

    for x, y in zip(test_pairs_lh,test_labels_lh) :
        lh_dict_test[x] = y

    for x, y in zip(test_pairs_lh_oracle,test_labels_lh_oracle) :
        lh_oracle_dict_test[x] = y
    print(len(lh_oracle_dict_train), len(lh_dict_train), len(lh_dict_test), len(lh_oracle_dict_test) )

    common_ids_train = {x:y for x, y in lh_oracle_dict_train.items() if x in train_pairs_lh}
    common_ids_test = {x:y for x, y in lh_oracle_dict_test.items() if x in test_pairs_lh}


    return common_ids_train,common_ids_test, train_pairs_lh, train_pairs_lh_oracle

def get_ecb_generated_images(common_ids_test,lh_overlap=True):
    rand_arg = {}
    rand_paired = {}
    main_path = os. getcwd()
    ecb_paired_train = torch.load(main_path + '/ViT/paired_train')
    ecb_paired_dev = torch.load(main_path + '/ViT/paired_dev')
    ecb_paired_test = torch.load(main_path + '/ViT/paired_test')

    ecb_paired_train_extra = torch.load(main_path + '/first_pair_ViT/first_pair_train')
    ecb_paired_dev_extra = torch.load(main_path + '/first_pair_ViT/first_pair_dev')
    ecb_paired_test_extra = torch.load(main_path + '/first_pair_ViT/first_pair_test')

    ecb_arg_train = torch.load(main_path + '/ViT/train')
    ecb_arg_dev = torch.load(main_path + '/ViT/dev')
    ecb_arg_test = torch.load(main_path + '/ViT/test')

    ecb_paired_train.update(ecb_paired_train_extra)
    ecb_paired_dev.update(ecb_paired_dev_extra)
    ecb_paired_test.update(ecb_paired_test_extra)

    if lh_overlap:
        ecb_paired_test = {(x.split("+")[0], x.split("+")[1]):y for x, y in ecb_paired_test.items()}
        len(ecb_paired_train), len(ecb_paired_dev), len(ecb_paired_test), len(ecb_arg_train),len(ecb_arg_dev), len(ecb_arg_test) #From embedding dictionary
        ecb_paired_test = {k:v for k,v in ecb_paired_test.items() if k in common_ids_test.keys()}
        ecb_paired_test = {x[0]+'+'+x[1]:y for x, y in ecb_paired_test.items()}
        print(len(ecb_paired_test))
    #create a new dict with random embeddings
    # ecb_paired_train = torch.load(main_path + '/SWIN/paired_train')
    # ecb_paired_dev = torch.load(main_path + '/SWIN/paired_dev')
    # ecb_paired_test = torch.load(main_path + '/SWIN/paired_test')

    # ecb_paired_train_extra = torch.load(main_path + '/first_pair_SWIN/first_pair_train')
    # ecb_paired_dev_extra = torch.load(main_path + '/first_pair_SWIN/first_pair_dev')
    # ecb_paired_test_extra = torch.load(main_path + '/first_pair_SWIN/first_pair_test')

    # ecb_arg_train = torch.load(main_path + '/SWIN/train')
    # ecb_arg_dev = torch.load(main_path + '/SWIN/dev')
    # ecb_arg_test = torch.load(main_path + '/SWIN/test')

    # ecb_paired_train = torch.load(main_path + '/BeIT/paired_train')
    # ecb_paired_dev = torch.load(main_path + '/BeIT/paired_dev')
    # ecb_paired_test = torch.load(main_path + '/BeIT/paired_test')

    # ecb_paired_train_extra = torch.load(main_path + '/first_pair_BeIT/first_pair_train')
    # ecb_paired_dev_extra = torch.load(main_path + '/first_pair_BeIT/first_pair_dev')
    # ecb_paired_test_extra = torch.load(main_path + '/first_pair_BeIT/first_pair_test')

    # ecb_arg_train = torch.load(main_path + '/BeIT/train')
    # ecb_arg_dev = torch.load(main_path + '/BeIT/dev')
    # ecb_arg_test = torch.load(main_path + '/BeIT/test')

    # ecb_paired_train = torch.load(main_path + '/Clip_paired_embeddings/paired_train')
    # ecb_paired_dev = torch.load(main_path + '/Clip_paired_embeddings/paired_dev')
    # ecb_paired_test = torch.load(main_path + '/Clip_paired_embeddings/paired_test')

    # ecb_arg_train = torch.load(main_path + '/Clip/train')
    # ecb_arg_dev = torch.load(main_path + '/Clip/dev')
    # ecb_arg_test = torch.load(main_path + '/Clip/test')




#     len(ecb_paired_train), len(ecb_paired_dev), len(ecb_paired_test), len(ecb_arg_train),len(ecb_arg_dev), len(ecb_arg_test) #From embedding dictionary
    return ecb_paired_train, ecb_paired_dev, ecb_paired_test, ecb_arg_train, ecb_arg_dev, ecb_arg_test


def generate_linear_maps(dataset,split, heu, dataset_folder, v_model = "vit", image_type= None, lh_overlap = False):
    heu = 'lh_oracle'
    lm_embedding = pickle.load(open(dataset_folder + f'/dpos_longformer/{split}_{heu}_LMembed.pkl', 'rb'))
    v_embedding = pickle.load(open(dataset_folder + f'/dpos_{v_model}_zeroshot/{split}_visiontensor_{v_model}.pkl', 'rb'))

    if image_type == "real":
        print(f"returning {image_type} {dataset} embeddings")
        v_embedding = pickle.load(open(dataset_folder + f'/dpos_{v_model}_zeroshot/{split}_visiontensor_{v_model}_{image_type}.pkl', 'rb'))

    def generate(v_embedding, lm_embedding):
        """
        Gets the Lin-Sem bidirectional linear maps of between image and text embeddings

        Parameters
        ----------
        v_embedding : tensor
            Embeddings from the various image models
        lm_embedding : tensor
            Embeddings from the text-only model (Longformer)

        Returns
        -------
        Tuple: Linear mapped embeddings between the two modalities in both directions
        """
        if lh_overlap: #to find the overlap between lh and lh oracle
            heu = 'lh'
        print("overlap heuristic", heu)
        file_path_1 = dataset_folder + f"/dpos_{v_model}_zeroshot/{split}_{heu}_coef_v_lm.npy" # vision-->text coefficients for train set
        file_path_2 = dataset_folder + f"/dpos_{v_model}_zeroshot/{split}_{heu}_coef_lm_v.npy" # text-->vision coefficients for train set
        print(file_path_1)
        if os.path.exists(file_path_1) and os.path.exists(file_path_2):
            print("file exists")
            mapper_v_lm = np.load(dataset_folder + f"/dpos_{v_model}_zeroshot/{split}_lh_oracle_coef_v_lm.npy")
            mapper_lm_v = np.load(dataset_folder + f"/dpos_{v_model}_zeroshot/train_lh_oracle_coef_lm_v.npy")
            if image_type == "real":
                mapper_v_lm = np.load(dataset_folder + f"/dpos_{v_model}_zeroshot/{split}_lh_oracle_coef_v_lm_{image_type}.npy")
                mapper_v_lm = np.load(dataset_folder + f"/dpos_{v_model}_zeroshot/{split}_lh_oracle_coef_lm_v_{image_type}.npy")

            print(f"returning {image_type} {dataset} {v_model} existing files")
            return mapper_v_lm, mapper_lm_v

        v_embedding = np.array(v_embedding)
        lm_embedding = np.array(lm_embedding)

        print(v_embedding.shape, lm_embedding.shape)
        #assert v_embedding.shape != lm_embedding.shape, f"Shapes of image and text embeddings must be same"
        mapper_v_lm = sklearn.linear_model.Ridge(fit_intercept=False).fit(v_embedding,lm_embedding)
        mapper_v_lm = mapper_v_lm.coef_

        mapper_lm_v = sklearn.linear_model.Ridge(fit_intercept=False).fit(lm_embedding,v_embedding)
        mapper_lm_v = mapper_lm_v.coef_

        np.save(dataset_folder + f"/dpos_{v_model}_zeroshot/{split}_{heu}_coef_v_lm_{image_type}", mapper_v_lm)
        np.save(dataset_folder + f"/dpos_{v_model}_zeroshot/{split}_{heu}_coef_lm_v_{image_type}", mapper_lm_v)

        print(f"saving Lin-Sem generated linear map coefficients for {dataset} with overlap {lh_overlap}")

        return mapper_v_lm, mapper_lm_v

    if lh_overlap:
        common_pairs, _, train_pairs_lh, train_pairs_lh_oracle = find_lh_lh_oracle_overlap(dataset, dataset_folder)
        v_embedding_dict = {x:y for x, y in zip(train_pairs_lh_oracle, v_embedding)}
        lm_embedding_dict = {x:y for x, y in zip(train_pairs_lh_oracle, lm_embedding)}

        v_embedding_dict = {k: v for k, v in v_embedding_dict.items() if k in common_pairs.keys()}
        lm_embedding_dict = {k: v for k, v in lm_embedding_dict.items() if k in common_pairs.keys()}

        v_embedding =list(v_embedding_dict.values())
        lm_embedding = list(lm_embedding_dict.values())
        v_embedding =  torch.stack(v_embedding, dim = 0)
        lm_embedding =  torch.stack(lm_embedding, dim = 0)

        print("size of overlap embeddings ",v_embedding.size(), lm_embedding.size())
#         v_embedding = np.array(v_embedding)
#         lm_embedding = np.array(lm_embedding)
        #print("shapes", v_embedding.shape, lm_embedding.shape)

    return generate(v_embedding, lm_embedding)

def vision_tokenize_new(tokenizer, mention_pairs,vision_map,vision_map_paired,  mention_map, m_end, max_sentence_len=1024, text_key='bert_doc', truncate=True):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'
    vision_fulltensor = []

    for (m1, m2) in mention_pairs:
        sentence_a = mention_map[m1][text_key]
        sentence_b = mention_map[m2][text_key]

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]), \
                   ' '.join([doc_start, sent_b, doc_end])

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

        #visiontensor_a = vision_map[m1]
        visiontensor_a = torch.tensor(vision_map[m1] ).unsqueeze(dim = 0)

        #print("vit a", visiontensor_a.size())
        #visiontensor_b = vision_map[m2]
        visiontensor_b = torch.tensor(vision_map[m2] ).unsqueeze(dim = 0)
        #print("vit b", visiontensor_b.size())

        paired_mid = m1+ '+' + m2
        paired_tensor= torch.tensor(vision_map_paired[paired_mid]).unsqueeze(dim = 0)

        #print("paired tensor size", paired_tensor.size())
        #visiontensor = torch.cat([visiontensor_a, visiontensor_a, visiontensor_b, visiontensor_a * visiontensor_b], dim=1)
        visiontensor = torch.cat([paired_tensor, visiontensor_a, visiontensor_b, visiontensor_a * visiontensor_b], dim=1)
        #print("vision full tensor size", visiontensor.size())
        vision_fulltensor.append(visiontensor)


    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = input_id[curr_start_index: m_end_index] + \
                           input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
            input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances, vision_tensor):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))
        v_tensor  = torch.cat(vision_tensor, dim=0)
        #print("vit full", vision_fulltensor.size())

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab,
                             'vision_tensor': v_tensor
                             }

        return tokenized_ab_dict

    if truncate:
        tokenized_ab  = ab_tokenized(pairwise_bert_instances_ab, vision_fulltensor)
        tokenized_ba = ab_tokenized(pairwise_bert_instances_ba, vision_fulltensor)
    else:
        instances_ab = [' '.join(instance) for instance in pairwise_bert_instances_ab]
        instances_ba = [' '.join(instance) for instance in pairwise_bert_instances_ba]
        tokenized_ab = tokenizer(list(instances_ab), add_special_tokens=False, padding=True)

        tokenized_ab_input_ids = torch.LongTensor(tokenized_ab['input_ids'])

        tokenized_ab = {'input_ids': torch.LongTensor(tokenized_ab['input_ids']),
                         'attention_mask': torch.LongTensor(tokenized_ab['attention_mask']),
                         'position_ids': torch.arange(tokenized_ab_input_ids.shape[-1]).expand(tokenized_ab_input_ids.shape)}

        tokenized_ba = tokenizer(list(instances_ba), add_special_tokens=False, padding=True)
        tokenized_ba_input_ids = torch.LongTensor(tokenized_ba['input_ids'])
        tokenized_ba = {'input_ids': torch.LongTensor(tokenized_ba['input_ids']),
                        'attention_mask': torch.LongTensor(tokenized_ba['attention_mask']),
                        'position_ids': torch.arange(tokenized_ba_input_ids.shape[-1]).expand(tokenized_ba_input_ids.shape)}

    return tokenized_ab, tokenized_ba

def predict_with_linsem(mention_map, model_name, linear_weights_path, test_pairs, text_key='bert_doc', max_sentence_len=1024, long=True, image_type = None, v_model=None, lin_sem_direction=None ):
    image_type = image_type
    v_model= v_model
    lin_sem_direction= lin_sem_direction
    dataset = 'ecb'
    dataset_folder = f'./datasets/{dataset}/'
    device = torch.device('cuda:1')
    #device_ids = list(range(1))
    device_ids = [1]
    linear_weights = torch.load(linear_weights_path,  map_location=torch.device('cpu'))
    scorer_module = Lin_Sem_Predictor(is_training=False, model_name=model_name, long=True,
                                      linear_weights=linear_weights,linear_map=True,zero_shot = True, heu = "lh", image_type = image_type, v_model = v_model, lin_sem_direction = lin_sem_direction).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer
    # prepare data, get the vision tokenization function here
    _,common_ids_test,_,_  = find_lh_lh_oracle_overlap(dataset, dataset_folder)
    ecb_paired_train, ecb_paired_dev, ecb_paired_test, ecb_arg_train, ecb_arg_dev, ecb_arg_test = get_ecb_generated_images(common_ids_test,lh_overlap=True)
    print(len(ecb_paired_train), len(ecb_paired_dev), len(ecb_paired_test), len(ecb_arg_train),len(ecb_arg_dev), len(ecb_arg_test))
    test_ab, test_ba = vision_tokenize_new(tokenizer, test_pairs,ecb_arg_test,ecb_paired_test,mention_map, parallel_model.module.end_id, text_key=text_key, max_sentence_len=max_sentence_len)
    scores_ab, scores_ba = predict_with_tp_fp_model(parallel_model, test_ab, test_ba, device, batch_size=5)
    #Lm_embed = predict_dpos(parallel_model, test_ab, test_ba, device, batch_size=128)
    return scores_ab, scores_ba, test_pairs

def predict_with_tp_fp_model(parallel_model, dev_ab, dev_ba, device, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    LM_embed = []
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices)
            #embed_ab = forward_ab(parallel_model, dev_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            #print("scores", scores_ab, scores_ba)
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())
    return torch.cat(all_scores_ab), torch.cat(all_scores_ba)

def save_dpos_scores_with_linsem(dataset, split, dpos_folder, heu='lh', threshold=0.999, text_key='bert_doc', max_sentence_len=1024, long=True, image_type = None, v_model=None, lin_sem_direction=None):
    image_type = image_type
    v_model= v_model
    lin_sem_direction= lin_sem_direction

    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    split ='test'
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
    #select the overlapping test pairs
    _,common_ids_test,_,_  = find_lh_lh_oracle_overlap(dataset, dataset_folder)
    test_pairs_dict = {pair:value for pair,value in zip(test_pairs, test_labels) if pair in common_ids_test.keys()}
    test_pairs = list(test_pairs_dict.keys())
    test_labels = list(test_pairs_dict.values())
    test_pairs = test_pairs[0:10] #unit testing with a small sample
    test_labels = test_labels[0:10]
    print("lh test pairs and labels",len(test_pairs ), len(test_labels))
    linear_weights_path = dpos_folder + "/linear.chkpt"
    bert_path = dpos_folder + '/bert'
    scores_ab, scores_ba, pairs = predict_with_linsem(evt_mention_map, bert_path, linear_weights_path, test_pairs, text_key, max_sentence_len, long=False, image_type = image_type, v_model=v_model, lin_sem_direction=lin_sem_direction)

    pickle.dump(test_pairs, open(dataset_folder + f'/dpos_{v_model}_zeroshot/{split}_{heu}_pairs.pkl', 'wb'))
    pickle.dump(scores_ab, open(dataset_folder + f'/dpos_{v_model}_zeroshot/{split}_{heu}_scores_ab_{lin_sem_direction}_{image_type}.pkl', 'wb'))
    pickle.dump(scores_ba, open(dataset_folder + f'/dpos_{v_model}_zeroshot/{split}_{heu}_scores_ba_{lin_sem_direction}_{image_type}.pkl', 'wb'))
    return scores_ab, scores_ba, pairs


if __name__ == '__main__':
    dataset = 'ecb'
    split = 'train'
    heu = 'lh_oracle'
    dataset_folder = f'./datasets/{dataset}'
    models = ['vit', 'SWIN', 'BEIT', 'CLIP']
    lin_sem_direction = ['lm_v', 'v_lm']
    image_type = ['generated']
    for im_type in image_type:
        for model in models:
            for lm_direction in lin_sem_direction:
                print(f"image type {im_type}, for model :{model} and for Linsem direction: {lm_direction}")
                scores_ab, scores_ba, pairs = save_dpos_scores_with_linsem(dataset, split, dpos_path, heu='lh', threshold=0.999, text_key='bert_doc', max_sentence_len=1024, long=True, image_type = im_type, v_model=model, lin_sem_direction=lm_direction)
