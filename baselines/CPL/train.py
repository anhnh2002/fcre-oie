import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment, gen_data
from encoder import EncodingModel
from sklearn.metrics import f1_score


class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto
        

    def train_model(self, encoder, training_data, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                loss = self.moment.contrastive_loss(hidden, labels, is_memory)
                if torch.isnan(loss):
                    print('loss is nan')
                    loss = 0.0
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update moment
                if is_memory:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        print('')   

    def f1(self, preds, labels):
        unique_labels = set(preds + labels)
        if self.config.na_id in unique_labels:
            unique_labels.remove(self.config.na_id)
        
        unique_labels = list(unique_labels)

        # Calculate F1 score for each class separately
        f1_per_class = f1_score(preds, labels, average=None)

        # Calculate micro-average F1 score
        f1_micro = f1_score(preds, labels, average='micro', labels=unique_labels)

        # Calculate macro-average F1 score
        # f1_macro = f1_score(preds, labels, average='macro', labels=unique_labels)

        # Calculate weighted-average F1 score
        f1_weighted = f1_score(preds, labels, average='weighted', labels=unique_labels)

        print("F1 score per class:", dict(zip(unique_labels, f1_per_class)))
        print("Micro-average F1 score:", f1_micro)
        # print("Macro-average F1 score:", f1_macro)
        print("Weighted-average F1 score:", f1_weighted)

        return f1_micro          

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 48
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        preds = []
        labels = []
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            preds.extend(pred)
            labels.extend(label.cpu().tolist())
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            # sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
            #     .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            # sys.stdout.flush()        
        print('')
        return self.f1(preds, labels)

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset


    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number
        # cur_acc, total_acc = [], []
        # cur_acc_num, total_acc_num = [], []
        cur_acc_wo_na, total_acc_wo_na = [], []
        cur_acc_num_wo_na, total_acc_num_wo_na = [], []

        cur_acc_w_na, total_acc_w_na = [], []
        cur_acc_num_w_na, total_acc_num_w_na = [], []

        memory_samples = {}
        data_generation = []
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations) in enumerate(sampler):
      
            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize)

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])

            # Data gen
            if self.config.gen == 1:
                gen_text = []
                for rel in current_relations:
                    for sample in memory_samples[rel]:
                        sample_text = self._get_sample_text(self.config.training_data, sample['index'])
                        gen_samples = gen_data(self.r2desc, self.rel2id, sample_text, self.config.num_gen, self.config.gpt_temp, self.config.key)
                        gen_text += gen_samples
                for sample in gen_text:
                    data_generation.append(sampler.tokenize(sample))
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True) 
                self.train_model(encoder, memory_data_initialize, is_memory=True)

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # Eval current task and history task
            # test_data_initialize_cur, test_data_initialize_seen = [], []
            # for rel in current_relations:
            #     test_data_initialize_cur += test_data[rel]
            # for rel in seen_relations:
            #     test_data_initialize_seen += historic_test_data[rel]
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])
            # ac1 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_cur)
            # ac2 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen)
            # cur_acc_num.append(ac1)
            # total_acc_num.append(ac2)
            # cur_acc.append('{:.4f}'.format(ac1))
            # total_acc.append('{:.4f}'.format(ac2))
            # print('cur_acc: ', cur_acc)
            # print('his_acc: ', total_acc)
            # Eval current task and history task wo na
            test_data_initialize_cur_wo_na, test_data_initialize_seen_wo_na = [], []
            for rel in current_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_cur_wo_na += test_data[rel]
                
            for rel in seen_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_seen_wo_na += historic_test_data[rel]

            ac1_wo_na = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_cur_wo_na)
            ac2_wo_na = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen_wo_na)

            # Eval current task and history task w na
            test_data_initialize_cur_w_na, test_data_initialize_seen_w_na = [], []
            for rel in current_relations:
                test_data_initialize_cur_w_na += test_data[rel]
                
            for rel in seen_relations:
                test_data_initialize_seen_w_na += historic_test_data[rel]
            
            ac1_w_na = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_cur_w_na)
            ac2_w_na = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen_w_na)

            # wo na
            cur_acc_num_wo_na.append(ac1_wo_na)
            total_acc_num_wo_na.append(ac2_wo_na)
            cur_acc_wo_na.append('{:.4f}'.format(ac1_wo_na))
            total_acc_wo_na.append('{:.4f}'.format(ac2_wo_na))
            print('cur_acc_wo_na: ', cur_acc_wo_na)
            print('his_acc_wo_na: ', total_acc_wo_na)

            # w na
            cur_acc_num_w_na.append(ac1_w_na)
            total_acc_num_w_na.append(ac2_w_na)
            cur_acc_w_na.append('{:.4f}'.format(ac1_w_na))
            total_acc_w_na.append('{:.4f}'.format(ac2_w_na))
            print('cur_acc_w_na: ', cur_acc_w_na)
            print('his_acc_w_na: ', total_acc_w_na)


        torch.cuda.empty_cache()
        return total_acc_num_wo_na, total_acc_num_w_na


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.na_id = 80
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.na_id = 41
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list_wo_na = []
    acc_list_w_na = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        acc_wo_na, acc_w_na = manager.train()
        acc_list_wo_na.append(acc_wo_na)
        acc_list_w_na.append(acc_w_na)
        torch.cuda.empty_cache()
    
    # wo na
    accs_wo_na = np.array(acc_list_wo_na)
    ave_wo_na = np.mean(accs_wo_na, axis=0)
    std_wo_na = np.std(accs_wo_na, axis=0)
    print('----------END')
    print('his_acc mean_wo_na: ', np.around(ave_wo_na, 4))
    print('his_acc std_wo_na: ', np.around(std_wo_na, 4))

    # w na
    accs_w_na = np.array(acc_list_w_na)
    ave_w_na = np.mean(accs_w_na, axis=0)
    std_w_na = np.std(accs_w_na, axis=0)
    print('his_acc mean_w_na: ', np.around(ave_w_na, 4))
    print('his_acc std_w_na: ', np.around(std_w_na, 4))




            
        
            
            


