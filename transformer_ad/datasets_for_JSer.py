import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.cuda
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import gc
import random

class MonitorEntityDataset_JSer(Dataset):
    def __init__(self, trainfiles, testfiles, sequence_length, z_dim, period=288, use_sfa=True, no_longterm=False, largest_bin=8, seed=0, is_minmax=True, gpu=None):
        
        self.gpu = gpu
        self.sequence_length = sequence_length
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')
        
        self.is_train = True
        self.train_index = []
        
        self.current_test = None
        self.test_index = {}
        self.curves = {}
        self.representative_curves = {}
        
        self.fea_dims = []
        
        self.use_sfa = use_sfa
        self.sfa = {}
        
        self.tensors = []
        
        data_reserved_for_sfa = []
        
        trainfiles = sorted(trainfiles)
        testfiles = sorted(testfiles)
        
        for i in range(len(trainfiles)):
            example_train = pd.read_csv(trainfiles[i])
            example_test = pd.read_csv(testfiles[i])
            
            self.fea_dims.append(example_train.shape[1])
            
            if is_minmax:
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            example_train = pd.DataFrame(scaler.fit_transform(example_train))
            example_test = pd.DataFrame(scaler.transform(example_test))
            
            len_test = len(example_test)
            
            example_all = pd.concat((example_train, example_test),axis=0)
            example_all.reset_index(drop=True, inplace=True)
            
            if no_longterm:
                tensor = np.expand_dims(example_all.values, axis = 2)
                tensor = tensor[period*2:]
            else:
                yester_1 = example_all.shift(period)
                yester_2 = example_all.shift(period*2)

                tensor = np.stack((example_all.values, yester_1.values, yester_2.values), axis = 2)
                tensor = tensor[period*2:]
            
            self.tensors.append(torch.from_numpy(tensor).float().to(self.device))
            self.train_index.extend([(i, j) for j in range(self.sequence_length-1, len(tensor) - len_test)])
            
            self.test_index[testfiles[i]] = [(i, j) for j in range(len(tensor)-len_test, len(tensor))]
            self.test_index[testfiles[i]+'_TRAINSET'] = [(i, j) for j in range(self.sequence_length-1, len(tensor) - len_test)]
            
            # data_reserved_for_sfa.append(example_train[-period:].values.swapaxes(0,1))
            len_train = len(example_train)
            seasons = len_train//period
            med = np.median(np.stack([example_train[len_train-period*(i+1):len_train-period*i].values for i in range(seasons)]),axis=0)
            self.curves[testfiles[i]] = [example_train[len_train-period*(i+1):len_train-period*i].values for i in range(seasons)]
            self.representative_curves[testfiles[i]] = med
            data_reserved_for_sfa.append(med.swapaxes(0,1))
                            

        self.sfa_dim = z_dim
        if use_sfa:
            # 计算sfa
            data_reserved_for_sfa=np.concatenate(data_reserved_for_sfa)
            
            paa_data = []
            paa_dim = 48
            for i in range(paa_dim):
                paa_data.append(np.mean(data_reserved_for_sfa[:,i*(period//paa_dim):(i+1)*(period//paa_dim)], axis=1))
                
            data_reserved_for_sfa = np.stack(paa_data, axis = 1)
            fft_data = np.fft.fft(data_reserved_for_sfa)
            
            #remove #0618 add pca
            #fft_data = np.concatenate((fft_data.real, fft_data.imag), axis = 1)
            #pca = PCA(n_components=z_dim)
            #fft_data = pca.fit_transform(fft_data)

            self.sfa_labels = [None for i in range(z_dim)]
            self.kmeans_models = [None for i in range(z_dim)]
            for i in range(0, z_dim):
                silhouette_avg = -1
                for j in range(2, largest_bin):
                    #0618 add pca
                    reduced_data = np.array([[x.real for x in fft_data[:,i]], [x.imag for x in fft_data[:,i]]]).swapaxes(0, 1)
                    #reduced_data = fft_data[:, i].reshape(-1, 1)

                    kmeans = KMeans(init="k-means++", n_clusters=j, n_init=4, random_state=seed)
                    kmeans.fit(reduced_data)

                    cluster_labels = kmeans.labels_

                    silhouette_avg_new = silhouette_score(reduced_data, cluster_labels)
                    if silhouette_avg_new > silhouette_avg:
                        silhouette_avg = silhouette_avg_new
                        #print(i,j)
                        #print('silhouette_avg:', silhouette_avg_new)
                        self.kmeans_models[i-1] = kmeans
                        
                        #remove #0618 add pca
                        for ind, rd in enumerate(reduced_data):
                            if (rd == np.array([0,0])).all():
                                cluster_labels[ind] = -1

                        self.sfa_labels[i-1] = cluster_labels
                    else:
                        continue

            self.sfa_labels = np.stack(self.sfa_labels, axis=1)
            self.enc = OneHotEncoder(sparse=False).fit(self.sfa_labels)

            self.sfa_labels = self.enc.transform(self.sfa_labels)

            self.sfa_dim = self.sfa_labels.shape[1]

            begin_loc=0
            for i in range(len(self.fea_dims)):
                self.sfa[i] = torch.from_numpy(self.sfa_labels[begin_loc:begin_loc+self.fea_dims[i]]).int().to(self.device)
                begin_loc += self.fea_dims[i]
        gc.collect()
         
            
    def train(self):
        self.is_train = True
        
    def test(self, current_test, use_trainset=False):
        self.is_train = False
        if use_trainset:
            self.current_test = current_test + '_TRAINSET'
        else:
            self.current_test = current_test
        
            
    def __len__(self):
        if self.is_train:
            return len(self.train_index)
        else:
            return len(self.test_index[self.current_test])
            

    def __getitem__(self, i):
        '''
        return metric_tensor of shape [longterm_size, fea_dim, seq_len]
            and sfa_tensor of shape [fea_dim, sfa_len]
        '''
        if self.use_sfa:
            if self.is_train:
                loc_set, loc_seq= self.train_index[i]
                return self.tensors[loc_set][loc_seq-self.sequence_length+1:loc_seq+1].transpose(0,2).contiguous(), self.sfa[loc_set]#.repeat(3, 1, 1)
            else:
                loc_set, loc_seq= self.test_index[self.current_test][i]
                return self.tensors[loc_set][loc_seq-self.sequence_length+1:loc_seq+1].transpose(0,2).contiguous(), self.sfa[loc_set]#.repeat(3, 1, 1)
        else:
            if self.is_train:
                loc_set, loc_seq= self.train_index[i]
                return self.tensors[loc_set][loc_seq-self.sequence_length+1:loc_seq+1].transpose(0,2).contiguous(), 0#.repeat(3, 1, 1)
            else:
                loc_set, loc_seq= self.test_index[self.current_test][i]
                return self.tensors[loc_set][loc_seq-self.sequence_length+1:loc_seq+1].transpose(0,2).contiguous(), 0#.repeat(3, 1, 1)
            
            

            
            
            
            
            
        
        
