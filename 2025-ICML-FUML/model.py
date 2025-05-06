import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class MembershipCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(MembershipCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return self.get_norm_outputs(h)
    
    def get_norm_outputs(self,inputs):
        norm = torch.norm(inputs, p=3, dim=1, keepdim=True) # p is adjustable, please refer appendix of our paper
        outputs = inputs / norm
        outputs = torch.relu(outputs) 

        return outputs

class net(nn.Module):
    def __init__(self, num_views, num_layer, dims, num_classes):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes  
        self.e = 0.00000000001
        dims = np.repeat(dims, num_layer, axis=1) 

        self.MembershipCollectors = nn.ModuleList([MembershipCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, data_list, label=None, test=False): 
            
        Weight, Membership, Credibility, Uncertainty, ConflictDegree = dict(), dict(), dict(), dict(), dict()

        if test and label == None:
            pass
        else:
            one_hot_labels = F.one_hot(label, self.num_classes)

        Weights, MMLogit, MMcrediblity = 0, 0, 0

        for view in range(self.num_views):
            Membership[view] = self.MembershipCollectors[view](data_list[view])
            Credibility[view] = self.get_test_credibility(Membership[view]) if test else self.get_train_credibility(Membership[view], one_hot_labels)            
            Uncertainty[view] = self.get_fuzzyUncertainty(Credibility[view])

        for view in range(self.num_views):
            conflictDegree = 0
            for v in range(self.num_views):
                if self.num_views > 1:
                    conflictDegree += self.get_ConflictDegree(Membership[view], Membership[v]) * (1/(self.num_views - 1)) 

            ConflictDegree[view] = conflictDegree
    
            Weight[view] =  (1 - Uncertainty[view]) * (1 - ConflictDegree[view]) + self.e  #避免出现0

        Weights = [Weight[key] for key in sorted(Weight.keys())]
        Weights = torch.stack(Weights)
        Weights = torch.softmax(Weights, dim=0)
        for view in range(self.num_views):
            MMLogit += Weights[view] * Membership[view]

        MMcrediblity = self.get_test_credibility(MMLogit) if test else self.get_train_credibility(MMLogit, one_hot_labels)
        MMuncertainty = self.get_fuzzyUncertainty(MMcrediblity)

        return Credibility, MMcrediblity, MMuncertainty 


    def get_train_credibility(self, predict, labels):
        top1Possibility = (predict*(1-labels)).max(1)[0].reshape([-1,1])
        labelPossibility = (predict*labels).max(1)[0].reshape([-1,1])
        neccessity = (1-labelPossibility)*(1-labels) + (1-top1Possibility)*labels
        conf = (predict + neccessity)/2
        return conf

    def get_test_credibility(self, membershipDegree): 
        if membershipDegree.shape[1] > 1:
            top2MembershipDegree = torch.topk(membershipDegree, k=2, dim=1, largest=True, sorted=True)[0]
            secMaxMembershipDegree = torch.where(membershipDegree == top2MembershipDegree[:,0].unsqueeze(1), top2MembershipDegree[:,1].unsqueeze(1), top2MembershipDegree[:,0].unsqueeze(1))
            confidence = (membershipDegree + 1 - secMaxMembershipDegree) / 2
        else:
            confidence = membershipDegree
        return confidence

    def get_fuzzyUncertainty(self, credibility):
        nonzero_indices = torch.nonzero(credibility)
        class_num = credibility.shape[1] 
        if len(nonzero_indices) > 1:
            H = torch.sum((-credibility*torch.log(credibility+self.e) - (1-credibility)*torch.log(1-credibility+self.e)), dim=1, keepdim=True)
            H = H / (class_num * torch.log(torch.tensor(2)))
        else:
            H = torch.tensor(0).unsqueeze(0)

        return H

    def get_ConflictDegree(self, vector1, vector2):
        distance = 1 - F.cosine_similarity(vector1, vector2, dim=1, eps=1e-8)
        distance = distance.view(-1, 1) 
        return distance 
    
                    

