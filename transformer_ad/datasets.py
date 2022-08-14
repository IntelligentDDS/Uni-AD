import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.cuda
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import gc
import copy

class MonitorEntityDataset(Dataset):
    def __init__(self, datafiles, sequence_length, z_dim, prefix='CTF_data/CTF_data/', len_test=23040, use_sfa=True, no_longterm=False, largest_bin=8, seed=0, is_iterate=True, gpu=None):
        
        self.gpu = gpu
        self.sequence_length = sequence_length
        self.datafiles = datafiles
        
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
        
        datafiles = sorted(datafiles)
        
        scaler = StandardScaler()
        mac_datafiles = []
        mac_id_in_tensors = 0
        for i in range(len(datafiles)):
            mac_id, mac_date = datafiles[i].split('_')
            mac_date, _ = mac_date.split('.')
            
            example = pd.read_csv(prefix + datafiles[i], header=None)
            example.interpolate(inplace=True)
            example.bfill(inplace=True)
            example.fillna(0, inplace=True)
            df = example.values
            
            if mac_date == '18':
                self.fea_dims.append(example.shape[1])
                
                if is_iterate:
                    old_mean_array = np.mean(df, axis=0, keepdims=True)
                    old_std_array = np.std(df, axis=0, keepdims=True)
                    #old_mean_array = np.median(df, axis=0, keepdims=True)
                    #old_std_array = np.median(np.abs(df-old_mean_array), axis=0, keepdims=True)

                    df = np.where(df > old_mean_array + 20 *old_std_array, old_mean_array + 20 *old_std_array, df)
                    df = np.where(df < old_mean_array - 20 *old_std_array, old_mean_array - 20 *old_std_array, df)

                    old_mean_array = np.mean(df, axis=0, keepdims=True)
                    old_std_array = np.std(df, axis=0, keepdims=True)
                    #old_mean_array = np.median(df, axis=0, keepdims=True)
                    #old_std_array = np.median(np.abs(df-old_mean_array), axis=0, keepdims=True)

                    df = np.where(df > old_mean_array + 20 *old_std_array, old_mean_array + 20 *old_std_array, df)
                    df = np.where(df < old_mean_array - 20 *old_std_array, old_mean_array - 20 *old_std_array, df)

                    df = (df - old_mean_array) / (old_std_array + 1e-9)
                    if np.any(sum(np.isnan(df)) != 0):
                        df = np.nan_to_num(df)
                    df[df > 20.0] = 20.0
                    df[df < -20.0] = -20.0
                    
                example_processed = pd.DataFrame(df)
                
                mac_datafiles = []
                mac_datafiles.append(example_processed)
            elif mac_date == '30':
                if is_iterate:
                    df = np.where(df > old_mean_array + 20 *old_std_array, old_mean_array + 20 *old_std_array, df)
                    df = np.where(df < old_mean_array - 20 *old_std_array, old_mean_array - 20 *old_std_array, df)

                    df = (df - old_mean_array) / (old_std_array + 1e-9)
                    if np.any(sum(np.isnan(df)) != 0):
                        df = np.nan_to_num(df)
                    df[df > 20.0] = 20.0
                    df[df < -20.0] = -20.0
                
                example_processed = pd.DataFrame(df)
                mac_datafiles.append(example_processed)
                
                example_all = pd.concat(mac_datafiles).reset_index(drop=True)
                
                if not is_iterate:
                    df = scaler.transform(example_all)
                    df[df > 20.0] = 20.0
                    df[df < -20.0] = -20.0
                    example_all = pd.DataFrame(df)
                
                if no_longterm:
                    period = len(example)
                    tensor = np.expand_dims(example_all.values, axis = 2)
                    tensor = tensor[period*2:]
                else:
                    period = len(example)
                    yester_1 = example_all.shift(period)
                    yester_2 = example_all.shift(period*2)

                    tensor = np.stack((example_all.values, yester_1.values, yester_2.values), axis = 2)
                    tensor = tensor[period*2:]
                
                self.tensors.append(torch.from_numpy(tensor).float().to(self.device))
                self.train_index.extend([(mac_id_in_tensors, j) for j in range(self.sequence_length-1, len(tensor) - len_test)])
                self.test_index[mac_id] = [(mac_id_in_tensors, j) for j in range(len(tensor)-len_test, len(tensor))]
                self.test_index[mac_id+'_TRAINSET'] = [(mac_id_in_tensors, j) for j in range(self.sequence_length-1, len(tensor) - len_test)]
                mac_id_in_tensors += 1
            else:
                if is_iterate:
                    df = np.where(df > old_mean_array + 20 *old_std_array, old_mean_array + 20 *old_std_array, df)
                    df = np.where(df < old_mean_array - 20 *old_std_array, old_mean_array - 20 *old_std_array, df)

                    mean_array = np.mean(df, axis=0, keepdims=True)
                    std_array = np.std(df, axis=0, keepdims=True)
                    #mean_array = np.median(df, axis=0, keepdims=True)
                    #std_array = np.median(np.abs(df-mean_array), axis=0, keepdims=True)

                    df = (df - old_mean_array) / (old_std_array + 1e-9)
                    if np.any(sum(np.isnan(df)) != 0):
                        df = np.nan_to_num(df)
                    df[df > 20.0] = 20.0
                    df[df < -20.0] = -20.0

                    old_mean_array = mean_array
                    old_std_array = std_array
                
                example_processed = pd.DataFrame(df)
                mac_datafiles.append(example_processed)
                
                if mac_date == '22':
                    #data_reserved_for_sfa.append(example_processed.values.swapaxes(0,1))
                    if is_iterate:
                        med = np.median(np.stack(mac_datafiles),axis=0)
                        self.representative_curves[mac_id] = med
                        data_reserved_for_sfa.append(med.swapaxes(0,1))
                    else:
                        scaler.fit(pd.concat(mac_datafiles).reset_index(drop=True))
                        processed_mac_datafiles = []
                        for f in mac_datafiles:
                            processed_mac_datafiles.append(scaler.transform(f))
                        med = np.median(np.stack(processed_mac_datafiles),axis=0)
                        self.representative_curves[mac_id] = med
                        self.curves[mac_id] = copy.deepcopy(processed_mac_datafiles)
                        data_reserved_for_sfa.append(med.swapaxes(0,1))
                            

        self.sfa_dim = z_dim
        if use_sfa:
            # 计算sfa
            data_reserved_for_sfa = np.concatenate(data_reserved_for_sfa)
            
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
                    #remove #0618 add pca
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
            
            

            
            
            
            
            
        
        
