import pickle
import torch.nn.functional as F
import torch

def toOneHot(env, actions):
    oneHot = torch.zeros(actions.size()[0], env.action_space.n) # [81, 1]
    oneHot[range(actions.size()[0]), actions.view(-1)] = 1 # [81, 4]
    return oneHot

def toIndexAction(env, oneHot):
    ac = torch.zeros(range(env.action_space.n)).view(1, -1) 
    ac = ac.expand(oneHot.size()[0], -1).contiguous().view(-1) 
    actions=ac[oneHot.view(-1)>0].view(-1)
    return actions
