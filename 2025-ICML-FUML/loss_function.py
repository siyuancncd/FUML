import torch.nn.functional as F

def get_loss(Credibility, MMcrediblity, target, num_classes):
    ce = lambda x, y: -(y * x.clamp_min(1e-7).log() + (1 - y) * (1 - x).clamp_min(1e-7).log()).sum(-1).mean()
    func = ce

    target = F.one_hot(target, num_classes)
    loss_total_acc = func(MMcrediblity, target)
    
    loss_acc = 0
    for key in Credibility.keys():
        loss_acc += func(Credibility[key], target)
        
    loss = loss_acc + loss_total_acc
        
    return loss