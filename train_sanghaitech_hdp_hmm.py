


import pandas  as pd 
import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
import numpy as np 
from torch import optim
from sklearn import metrics 
import os
import copy
cos = nn.CosineSimilarity(dim=2, eps=1e-6)
import sys
from gpytorch.kernels import ScaleKernel, SpectralMixtureKernel, RBFKernel, CosineKernel, MaternKernel, PiecewisePolynomialKernel, RQKernel, PolynomialKernelGrad
from hdp_hmm import StickyHDPHMM

dir_path ="SanghaiTech/Split/"
fc7_features_path = os.path.join(dir_path, 'fc7-features')
annotation_path = os.path.join(dir_path, 'annotations')
root_dir = "SanghaiTech/Videos/"
annotated_videos = os.listdir(os.path.join(root_dir, 'testing', 'fc7-features'))
unannotated_videos = os.listdir(os.path.join(root_dir, 'training', 'preprocessed/'))

def hdm_dmm(features, output, no_posterior_steps=100, out_th = 35):
    
    data_to_pass = features.data.cpu().numpy()
    stickyhdmm = StickyHDPHMM(data_to_pass)
    mean_output = torch.quantile(output, out_th, axis = 1)
    topk_output = torch.zeros_like(mean_output)
    for i in range(no_posterior_steps):
        stickyhdmm.sampler()
    cluster_numbers = np.array(stickyhdmm.state)
    cluster_numbers = torch.from_numpy(cluster_numbers).to(device)
    for i in range(len(features)):
        video_cluster = cluster_numbers[i]
        cluster_no_score = {}
        for j, cluster_no in enumerate(video_cluster):
            if output[i][j]<mean_output[i]:
                continue
            if cluster_no in cluster_no_score:

                cluster_no_score[cluster_no] = torch.max(cluster_no_score[cluster_no], output[i][j])
            else:
                cluster_no_score[cluster_no] = output[i][j]
        video_sum = torch.zeros_like(mean_output[0])
        for k, v in cluster_no_score.items():
            video_sum+=v
        topk_output[i] = video_sum/len(cluster_no_score)
    return topk_output

        
        

class CalibratedK(torch.nn.Module):
    def __init__(self):
        super(CalibratedK, self).__init__()
    def forward(self, abnormal_outputs, normal_outputs, abnormal_features, normal_features, sim_th, out_th, no_segments = 32):
     
        topk_output = hdm_dmm(abnormal_features,abnormal_outputs, no_posterior_steps=10, out_th = out_th)
        #normal_max_value = compute_topk(normal_features, normal_outputs, sim_th, out_th)
        [normal_max_value, _] = torch.max(normal_outputs, axis=1)
        hinge_loss = torch.zeros_like(abnormal_outputs)[0][0]
        for normal in normal_max_value:
            
            topk_loss = 1-topk_output+normal
            topk_loss[topk_loss<0]=0
            topk_loss = torch.sum(topk_loss)
            hinge_loss += topk_loss

        return hinge_loss/(normal_outputs.shape[0])  



class GCNConv(torch.nn.Module):
    def __init__(self, input_channels, out_channels):
        super(GCNConv, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Linear(input_channels, out_channels)#Parameter(FloatTensor(input_channels, out_channels))
        #self.reset_parameters()
    def reset_parameters(self):
        stdv = 1./sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = self.weight(input)#input.matmul(self.weight)
        output = adj.matmul(support)
        return output 

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_size = 32, no_segments =32, no_mixtures = 5, initialize = False, X=None, y=None):
        super(Net, self).__init__()
        self.gc1 = GCNConv(input_dim, 128)
        self.gc2= GCNConv(128, 64)
        self.gc3 = GCNConv(input_dim, 128)
        self.gc4= GCNConv(128, 64)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.6)
        
        self.lstm = torch.nn.LSTM(128,hidden_size,5,batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.hidden_size = hidden_size
        self.norm = torch.nn.BatchNorm1d(hidden_size)
        
        self.no_segments = no_segments
        self.covar_module1 = RBFKernel(ard_num_dims=input_dim)#SpectralMixtureKernel(num_mixtures=6,  ard_num_dims=input_dim, eps = 1e-06)
        self.covar_module2 = SpectralMixtureKernel(num_mixtures=6,  ard_num_dims=128, eps = 1e-06)
        
        if initialize:
            self.covar_module1.initialize_from_data(X, y)
            

    def compute_feature_adjancency(self, x, covar):
       
        adj = covar(x).evaluate()
        I = Variable(torch.eye(adj.shape[1]), requires_grad = True).to(device)
        I = I.repeat(x.shape[0], 1, 1)
        adj_til = adj+I 
        d_inv_sqrt2 = torch.diag_embed(torch.pow(torch.sum(adj_til, dim = 2), -0.5))
        adj_hat = d_inv_sqrt2.matmul(adj_til).matmul(d_inv_sqrt2)
        
        return adj_hat

    def compute_temporal_adjancency(self, x):
       adj = torch.zeros(x.shape[1], x.shape[1])
       for i in range(len(adj)):
            for j in range(len(adj)):
                adj[i][j] = torch.exp(-torch.abs(torch.tensor(abs(i-j))))
       adj = Variable(adj, requires_grad = True).to(device)
       adj = adj.repeat(x.shape[0], 1, 1)
       I = Variable(torch.eye(adj.shape[1]), requires_grad = True).to(device)
       I = I.repeat(x.shape[0], 1, 1)
       adj_til = adj+I 
       d_inv_sqrt2 = torch.diag_embed(torch.pow(torch.sum(adj_til, dim = 2), -0.5))
       adj_hat = d_inv_sqrt2.matmul(adj_til).matmul(d_inv_sqrt2)
     
        
       return adj_hat




    def forward(self, x):
        
       
        adj_1_feat = self.compute_feature_adjancency(x, self.covar_module1)
        
        x_feat_1 = self.gc1(x, adj_1_feat)
       

        adj_1_temp = self.compute_temporal_adjancency(x)
        x_temp_1 = self.gc3(x, adj_1_temp)
       

        x = x_feat_1+x_temp_1
        x = self.relu(x)
        x = self.dropout(x)

        x,_ = self.lstm(x)
        feat= self.norm(x)
        x = self.fc(feat)
        x = self.sigmoid(x)
        return [feat, x]

        
def get_output(X, model):
    X = torch.from_numpy(X)
    X = Variable(X).to(device)
    [_, output] = model(X.float())
    return output       


def getframeauc(model, X_test_abnormal, X_test_normal,  video_names_abnormal, video_names_normal):
    no_segments = X_test_abnormal.shape[1]
    predictions_abnormal = get_output(X_test_abnormal, model)
    predictions_normal = get_output(X_test_normal, model)
    predictions_abnormal = predictions_abnormal.data.cpu().numpy().flatten()
    predictions_normal = predictions_normal.data.cpu().numpy().flatten()
   
    predictions_abnormal = predictions_abnormal.reshape(len(X_test_abnormal),no_segments)
    predictions_normal = predictions_normal.reshape(len(X_test_normal), no_segments)
    GT, Pred = [], []
    clip_size = 16
    video_names = np.concatenate([video_names_abnormal, video_names_normal])
    predictions = np.concatenate([predictions_abnormal, predictions_normal])
    for i, video in enumerate(video_names):
        
        prediction = predictions[i]
        no_clips = len(sorted(os.listdir(fc7_features_path+"/testing/"+video)))
        thirty2_shots = np.round(np.linspace(0, no_clips-1, 33))
        p_c = 0
        clip_pred_score = np.zeros(no_clips)
        for ishots in range(0, len(thirty2_shots)-1):
            ss = int(thirty2_shots[ishots])
            ee = int(thirty2_shots[ishots+1])
           
            if ee<ss or ee==ss:
                clip_pred_score[ss] = prediction[p_c]
            else:
                
                clip_pred_score[ss:ee] = prediction[p_c]
            p_c+=1
        
        if video in annotated_videos:
            
            val = np.load(os.path.join(root_dir, 'testing', 'test_frame_mask', video+".npy"))
            number_frames = len(val)
            GT.extend(val.tolist())
        elif video in unannotated_videos:
           
            number_frames = len(os.listdir(os.path.join(root_dir, 'training', 'preprocessed', video)))
            val = np.zeros(number_frames)
            GT.extend(val.tolist())
        else:
            print("Unusual")
            print(video)
        frame_pred = np.zeros(number_frames)
        for j in range(no_clips):
            start_frame = j*clip_size
            if (j+1)*clip_size>number_frames:
                end_frame = number_frames
            else:
                end_frame = (j+1)*clip_size
            frame_pred[start_frame: end_frame] =clip_pred_score[j]
        Pred.extend(frame_pred.tolist())
    fpr, tpr, thresholds = metrics.roc_curve (GT, Pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc



if __name__=="__main__":
    [_, run,  out_th] = sys.argv


    X_train_abnormal, X_train_normal = np.load("Dataset/SanghaiTech/X_train_abnormal.npy",allow_pickle=True), np.load("Dataset/SanghaiTech/X_train_normal.npy",allow_pickle=True)
    X_test_abnormal, X_test_normal = np.load("Dataset/SanghaiTech/X_test_abnormal.npy"), np.load("Dataset/SanghaiTech/X_test_normal.npy")
    video_names_abnormal, video_names_normal = np.load("Dataset/SanghaiTech/videos_test_abnormal.npy"), np.load("Dataset/SanghaiTech/videos_test_normal.npy")

    #Training settings
    batch_size = 16
    lr = 0.01
    sim_th = float(35)/100
    out_th = float(out_th)/100
    hidden_size = 32
    no_segments = X_train_abnormal.shape[1]
    max_iterations = 50000
    input_dim = X_train_abnormal.shape[2]
    abnormal_idx = list(range(len(X_train_abnormal)))
    normal_idx = list(range(len(X_train_normal)))
    model = Net(input_dim=input_dim,hidden_size=hidden_size)
    customobjective = CalibratedK()
    model.to(device)
    customobjective.to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay = 0.0001)
    best_auc = 0
    aucs = []
    losses =[]
    for i in range(max_iterations+1): 

        model.train()
        np.random.shuffle(abnormal_idx)
        np.random.shuffle(normal_idx)
        # In each batch,half is positive and half is negative  
        train_abnormal_feat = X_train_abnormal[abnormal_idx[:int(batch_size/2)]]
        train_normal_feat = X_train_normal[normal_idx[:int(batch_size/2)]]
        train_feat = np.concatenate([train_abnormal_feat, train_normal_feat])
        train_feat = torch.from_numpy(train_feat)
        train_feat = Variable(train_feat, requires_grad = True).to(device)
        optimizer.zero_grad()
        [feats, outputs] = model(train_feat.float())
        outputs = outputs.squeeze()
        abnormal_outputs, normal_outputs = outputs[:int(batch_size/2)], outputs[int(batch_size/2):]
        abnormal_features = feats[:int(batch_size/2)]
        normal_features = feats[int(batch_size/2):]
        loss = customobjective(abnormal_outputs,normal_outputs,abnormal_features, normal_features, sim_th, out_th, no_segments = no_segments)
        loss.backward()
        a = loss.data.cpu()
        losses.append(a)
        optimizer.step()
        if i%10==0:
            model.eval()
            test_abnormal = torch.from_numpy(X_test_abnormal)
            test_abnormal = Variable(test_abnormal).to(device)
            [_, predictions_abnormal] = model(test_abnormal.float())
            predictions_abnormal = predictions_abnormal.reshape(-1, no_segments)
            predictions_abnormal = predictions_abnormal.data.cpu().numpy()
            test_normal = torch.from_numpy(X_test_normal)
            test_normal = Variable(test_normal).to(device)
            [_, predictions_normal] = model(test_normal.float())
            predictions_normal = predictions_normal.reshape(-1, no_segments)
            predictions_normal = predictions_normal.data.cpu().numpy()
            auc_score = getframeauc(model, X_test_abnormal, X_test_normal,  video_names_abnormal, video_names_normal)
            aucs.append(auc_score)
            
            if auc_score>best_auc:
                best_auc = auc_score
                print("Saving model")
                torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("models/SanghaiTech/model_hdm_hmm_"+str(run)+"_"+str(lr)+"_"+str(sim_th)+"_"+str(out_th)+".pth.tar"))
            print(" For the iteration", i, "Best AUC", best_auc)
               
    losses = np.array(losses)
    aucs = np.array(aucs)
    np.save("logs/SanghaiTech/auc_hdm_hmm_"+str(run)+"_"+str(lr)+"_"+str(sim_th)+"_"+str(out_th)+".npy", aucs)
    np.save("logs/SanghaiTech/losses_hdm_hmm_"+str(run)+"_"+str(lr)+"_"+str(sim_th)+"_"+str(out_th)+".npy", losses)

   
    





