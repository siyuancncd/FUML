import os
import random
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score

from options import get_dataloader, get_config
from data import *
from loss_function import get_loss
from model import net
# np.set_printoptions(precision=4, suppress=True)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train_test(args):
    device = args.device

    config, dataloader = get_config(args.dataset) 
    train_loader, test_loader = get_dataloader(args.dataset, args.conflictive_test) 

    model = net(dataloader.num_views, num_layer = config['layer_num'], dims=dataloader.dims, num_classes=dataloader.num_classes)
    model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    best_test_acc = 0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        
        for X, Y, indexes in train_loader:
            for v in range(dataloader.num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)

            Credibility, MMcrediblity, MMuncertainty  = model(X, Y, test=False)
            loss = get_loss(Credibility, MMcrediblity, Y, dataloader.num_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            num_correct, num_sample = 0, 0

            Y_pre_total = None
            Y_total = None
            MMuncertainty_total = None

            for X, Y, indexes in test_loader:
                for v in range(dataloader.num_views):
                    X[v] = X[v].to(device)
                Y = Y.to(device)

                with torch.no_grad():
                    Credibility, MMcrediblity, MMuncertainty = model(X, test=True)
                    
                    _, Y_pre = torch.max(MMcrediblity, dim=1)
                    num_correct += (Y_pre == Y).sum().item()
                    num_sample += Y.shape[0]
                
                Y_pre = np.array(Y_pre.cpu())    
                Y = np.array(Y.cpu())  
                MMcrediblity = np.array(MMcrediblity.cpu())
                MMuncertainty = np.array(MMuncertainty.cpu()) 

                if Y_pre_total is None:
                    Y_pre_total = Y_pre
                    Y_total = Y
                    MMuncertainty_total = MMuncertainty
                    MMcrediblity_total = MMcrediblity
                else:
                    Y_pre_total = np.hstack([Y_pre_total, Y_pre])
                    Y_total = np.hstack([Y_total, Y])
                    MMuncertainty_total = np.hstack([MMuncertainty_total.squeeze(), MMuncertainty.squeeze()])
                    MMcrediblity_total = np.vstack([MMcrediblity_total, MMcrediblity])
            
            acc = num_correct / num_sample
            F1 = f1_score(Y_total, Y_pre_total, average='macro')
            precision = precision_score(Y_total, Y_pre_total, average='macro')
            
            if acc > best_test_acc:
                best_test_acc = acc
                best_test_F1 = F1
                best_test_precision = precision

            if acc > best_test_acc:
                best_test_acc = acc
            print('Epoch:{:.0f} ====> best acc: {:.4f} acc: {:.4f} F1: {:.4f}  P: {:.4f} uncer: {:.4f}'.format(epoch,best_test_acc,acc,F1,precision,np.mean(MMuncertainty_total)))
            
    return best_test_acc, best_test_precision, best_test_F1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PIE', metavar='N',
                        help='dataset name') # PIE, Scene, LandUse, HW, NUSOBJ, Fashion, Leaves, MSRC
    parser.add_argument('--conflictive_test', type=bool, default=False, metavar='N',
                        help='conflicting or not')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='N',
                        help='gpu or cpu')
    parser.add_argument('--seed_list', type=int, default=[1,2,3,4,5,6,7,8,9,10], metavar='N',
                        help='random seed') 
    args = parser.parse_args()

    print("Processor: ", os.getpid())
    Acc_list = []
    P_list = []
    F_score_list = []
    for seed in args.seed_list:
        print("seed = ", seed)
        setup_seed(seed)
        Acc,P,F_score = train_test(args)
        Acc_list.append(Acc)
        P_list.append(P)
        F_score_list.append(F_score)

    print("***************************************")
    print("***************************************")
    print("Acc :", str(round(np.mean(Acc_list), 4)), " +- ", str(round(np.std(Acc_list), 4)) )
    print("P :", str(round(np.mean(P_list), 4)), " +- ", str(round(np.std(P_list), 4)) )
    print("F_score :", str(round(np.mean(F_score_list), 4)), " +- ", str(round(np.std(F_score_list), 4)) )
    print("***************************************")
    print("***************************************")