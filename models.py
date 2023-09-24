import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pyhocon
import os
from inspect import getfullargspec
from collections import defaultdict
import sklearn.linear_model

dataset_folder = f'./datasets/ecb'

#dataset_folder = f'./datasets/gvc'


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)
#this class instantiates the cross domain models, gets the linear maps between the embeddings and trains the MLP pairwise scoer using both the LM and the V->LM mapped features.

class Lin_Sem_Predictor(nn.Module):
    def __init__(self, is_training=True, long=True, model_name='allenai/longformer-base-4096',
                 linear_weights=None, linear_map = False, zero_shot = True, heu = "lh", image_type = "generated", v_model = "vit", lin_sem_direction = "v_lm"):
        super(Lin_Sem_Predictor, self).__init__()

        print("Initialized Lin-Sem based vision and text models")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.long = long
        self.linear_map = linear_map
        self.zero_shot = zero_shot
        self.heu = heu
        self.image_type = image_type
        self.v_model = v_model
        self.lin_sem_direction = lin_sem_direction


        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 8, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # this is for zero shot or direct replacement of LM trained scorer head with either vision features or V->LM mapped features
        self.zero_shot_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
            self.zero_shot_layer.apply(init_weights)
        else:
            #self.linear.load_state_dict(linear_weights)
            self.zero_shot_layer.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=None)
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2, v_tensor):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)
        lm_native = torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)



        #carry out the linear mapping here after getting the LM features paired

        if self.linear_map:
            device = torch.device('cuda:1')
            np_v_tensor = v_tensor.detach().cpu().numpy()
            np_lm_native = lm_native.detach().cpu().numpy()
            mapper = np.load(dataset_folder + f"/dpos_{self.v_model}_zeroshot/train_{self.heu}_coef_{self.lin_sem_direction}_{self.image_type}.npy")

            if self.lin_sem_direction == "v_lm":
                mapped_v_tensor = np_v_tensor @ mapper.transpose()
                mapped_tensor =torch.tensor(mapped_v_tensor).to(device)
            elif self.lin_sem_direction == "lm_v":
                mapped_lm_tensor = np_lm_native @ mapper.transpose()
                mapped_tensor =torch.tensor(mapped_lm_tensor).to(device)
            print(f"Appling inference with Lin-Sem generated coefficients for {self.v_model} {self.heu} {self.image_type} {self.lin_sem_direction}")

            return mapped_tensor


        else:
            #print("No linear maps carried out, training with both features")

            #return torch.cat([lm_native, v_tensor ], dim=1) if
            #return v_tensor
            print("getting LM native")
            return lm_native
            #full_tensor = torch.cat([lm_native, v_tensor ], dim=1)
            #print("full_tensor", full_tensor.size())

            #return torch.cat([lm_native, v_tensor ], dim=1)







    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False, v_tensor=None, zero_shot = False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2, v_tensor = v_tensor)
        if lm_only:
            return lm_output


        if self.zero_shot:
            print("getting zero shot and linear map")
            #print("getting mapped output for CLIp reverse ")

            #return self.zero_shot_layer(lm_output)
            return self.zero_shot_layer(lm_output.float())
            #return lm_output
        elif self.zero_shot==False:
            print("getting native LM ")
            return lm_output

class VLEncoder(nn.Module):
    def __init__(self, is_training=True, long=True, model_name='allenai/longformer-base-4096',
                 linear_weights=None, linear_map = False, zero_shot = True):
        super(VLEncoder, self).__init__()

        print("Initialized Vision Encoder model with Vision and LM features")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.long = long
        self.linear_map = linear_map
        self.zero_shot = zero_shot


        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 8, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # this is for zero shot or direct replacement of LM trained scorer head with either vision features or V->LM mapped features
        self.zero_shot_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
            self.zero_shot_layer.apply(init_weights)
        else:
            #self.linear.load_state_dict(linear_weights)
            self.zero_shot_layer.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=None)
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2, v_tensor):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)
        lm_native = torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)



        #carry out the linear mapping here after getting the LM features paired

        if self.linear_map:
            device = torch.device('cuda:1')
            np_v_tensor = v_tensor.detach().cpu().numpy()
            np_lm_native = lm_native.detach().cpu().numpy()

#             mapper_ = sklearn.linear_model.Ridge(fit_intercept=False).fit(np_v_tensor,np_lm_native)
#             mapper_ = mapper_.coef_
            #print("l map weights dimension", mapper_.shape)


            # Vision --> LM generated images
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_vit_zeroshot/train_lh_oracle_coef_v_lm.npy")
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_SWIN_zeroshot/train_lh_oracle_coef_v_lm.npy")
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_BEIT_zeroshot/train_lh_oracle_coef_v_lm.npy")
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_CLIP_zeroshot/train_lh_oracle_coef_v_lm.npy")


           # Vision --> LM actual images

            #mapper_v_lm = np.load(dataset_folder + f"/dpos_vit_zeroshot/train_lh_oracle_coef_v_lm_real.npy")

            #mapper_v_lm = np.load(dataset_folder + f"/dpos_SWIN_zeroshot/train_lh_oracle_coef_v_lm_real.npy")
            mapper_v_lm = np.load(dataset_folder + f"/dpos_BEIT_zeroshot/train_lh_oracle_coef_v_lm_real.npy")
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_CLIP_zeroshot/train_lh_oracle_coef_v_lm_real.npy")
            print("Inititializing linear projected scores from actual images Vision --> LM")


            # LM --> Vision actual images

            #mapper_v_lm = np.load(dataset_folder + f"/dpos_vit_zeroshot/train_lh_oracle_coef_lm_v_real.npy")
            #print("Inititializing linear projected scores from actual images LM --> Vision")
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_SWIN_zeroshot/train_lh_oracle_coef_lm_v_real.npy")
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_BEIT_zeroshot/train_lh_oracle_coef_lm_v_real.npy")
            #mapper_v_lm = np.load(dataset_folder + f"/dpos_CLIP_zeroshot/train_lh_oracle_coef_lm_v_real.npy")

            #print("Inititializing linear projected scores from actual images LM --> Vision")


            #mapped_v_tensor = np_v_tensor @ mapper_v_lm.transpose() #try the reverse mapping below

            # LM--> Vision

            #mapper_v_lm = np.load(dataset_folder + f"/dpos_vit_zeroshot/train_lh_oracle_coef_v_lm.npy") #getting the reverse map

            mapped_v_tensor = np_lm_native @ mapper_v_lm.transpose()

            print("Linear Projection done")
            mapped_v_tensor =torch.tensor(mapped_v_tensor).to(device)
            #print("mapped v tensor dimensions", mapped_v_tensor.size()) #batch size by 3072 i.e., all features

            print("linear map vision features only")
            #return torch.cat([lm_native, mapped_v_tensor ], dim=1)
            return mapped_v_tensor
#         all_mapped[i+m]  = mapped_
        else:
            #print("No linear maps carried out, training with both features")

            #return torch.cat([lm_native, v_tensor ], dim=1) if
            #return v_tensor
            print("getting LM native")
            return lm_native
            #full_tensor = torch.cat([lm_native, v_tensor ], dim=1)
            #print("full_tensor", full_tensor.size())

            #return torch.cat([lm_native, v_tensor ], dim=1)







    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False, v_tensor=None, zero_shot = False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2, v_tensor = v_tensor)
        if lm_only:
            return lm_output


        if self.zero_shot:
            print("getting zero shot and linear map")
            #print("getting mapped output for CLIp reverse ")

            #return self.zero_shot_layer(lm_output)
            return self.zero_shot_layer(lm_output.float())
            #return lm_output
        elif self.zero_shot==False:
            print("getting native LM ")
            return lm_output

#             print("training with both lm and vision features")
#             print("lm output",lm_output.size() )


            #return self.linear(lm_output)



        #return self.linear(lm_output)
class CrossEncoder_cossim(nn.Module):
    def __init__(self, is_training=True, long=True, model_name='allenai/longformer-base-4096',
                 linear_weights=None):
        super(CrossEncoder_cossim, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=None)
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)


        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_sim = cos(arg1_vec,  arg2_vec)
        print("getting cosine similarities")
        return cosine_sim

        #return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        if lm_only:
            return lm_output

        return lm_output
        #return self.linear(lm_output)





class CrossEncoder(nn.Module):
    def __init__(self, is_training=True, long=True, model_name='allenai/longformer-base-4096',
                 linear_weights=None):
        super(CrossEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                global_attention_mask=None)
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)

        return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        if lm_only:
            return lm_output

        return self.linear(lm_output)
