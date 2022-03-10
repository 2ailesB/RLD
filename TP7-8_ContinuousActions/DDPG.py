import random
from datetime import datetime
import yaml
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from memory import *
from core import *
from utils import *
import torch
import gridworld
import gym
import argparse
import sys
import matplotlib
import copy
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")


class DDPG(object):
    """
        class that defines the agent using DDPG to play
    """

    def __init__(self, env, opt):
        # environnement parameters
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        # Now with continuous actions, the first elt is the lowest acepted value and the second the highest
        self.action_space = env.action_space
        # number of possible actions
        self.action_space_n = self.action_space.shape[0]
        self.obs_space = env.observation_space
        self.featureExtractor = opt.featExtractor(env)
        self.test = False
        self.nbEvents = 0

        # hyper-parameters
        self.gamma = opt.gamma  # parameter for computation of y

        # Memory
        self.mem_size = opt.mem_size  # mémoire de batches qui contient 10000 transitions
        self.mbs = opt.mbs  # taille des minibatch pour l'apprentissage /!\ mbs <=mem_size
        self.memory = Memory(self.mem_size)  # Memory
        self.norm_rew = opt.norm_rew

        # Networks
        # Q doit dépendre des actions et des état ==> concaténer + modifier le réseau
        self.Qvalue = NN(self.featureExtractor.outSize + self.action_space_n, 1, layers=opt.ValueLayers, finalActivation=None, activation=F.leaky_relu, dropout=0.)  # first NN : learn
        self.QvalueHat = copy.deepcopy(self.Qvalue)  # second NN : play
        self.policy = NN(self.featureExtractor.outSize, self.action_space_n, layers=opt.ActorLayers, finalActivation=F.tanh, activation=F.leaky_relu, dropout=0.)  # first NN : learn
        self.policyHat = copy.deepcopy(self.policy)  # second NN : play
        self.lrValue = opt.ValueLr  # learning rate
        self.lrPolicy = opt.ActorLr
        self.C = opt.C  # update target each C steps
        self.evts = 0  # evts
        self.freqOptim = opt.freqOptim
        self.freqVerbose = opt.freqVerbose
        self.QvalueOptimizer = torch.optim.Adam(
            self.Qvalue.parameters(), lr=self.lrValue)
        self.policyOptimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lrPolicy)
        self.QvalueLoss = F.smooth_l1_loss
        self.policyLoss = F.smooth_l1_loss
        self.noise = Orn_Uhlen(self.action_space_n, mu=opt.mu, sigma=opt.sigma)
        self.tau = opt.tau

    # en test on prend le max, en apprentissage on fait un epsilon greddy pour explorer au hasard de temps en temps
    def act(self, obs):
        """
            Find action by sampling from the probability of the actor + a chosen noise
            Input :
                obs : the state of the agent
            Output : 
                pol : The actions to be played
        """

        # with torch.no_grad():
        # convert obs to torch object
        tobs = torch.from_numpy(obs).float()

        # add noise
        # epsilon = np.random.normal(0, 1, tobs.shape[0]) # observations in R^(tobs.shape[0])
        # epsilon = torch.from_numpy(epsilon)
        epsilon = self.noise.sample()

        pol = self.policyHat.forward((tobs).float())
        if not self.test:  # En test on n'explore plus donc on n'ajoute pas le espilon
            pol += epsilon

        minval = torch.from_numpy(self.action_space.low)
        maxval = torch.from_numpy(self.action_space.high)
        pol = pol.clamp(minval, maxval)
        
        return pol

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    def soft_update(self, source_model, target_net, tau):
        """
            Compute a soft update of the target network from the source model
            Input : 
                source_model (torch net) : the source model to copy from
                target_model (torch net) : the target model to copy to
                tau (float) : the trade off between the two networks 
                    (0 <= tau <= 1, 0 being no update (we keep only target params), 1 being complete update)
            Output : 
        """

        for target_param, source_param in zip(target_net.parameters(), source_model.parameters()):
            target_param.data.copy_(
                tau*source_param.data + (1.0-tau)*target_param.data)

    # apprentissage de l'agent
    def learn(self):

        """
            Learning part of the algorithm
            Input : 
            Output : 
                valuel : the loss of the Value function
                -policyl : the loss of the policy
        """
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return self, 0, 0

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if self.memory.nentities <= self.mbs:
            print('memory not enough filled ', self.memory.nentities,
                  'available of out ', self.mbs, 'needed for 1 batch')
            return self, 0, 0

        # sinon on calcule y et on fait une descente de gradient
        # On tire aléatoirement un mini batch de transitions
        chosenIdx, z, chosentr = self.memory.sample(self.mbs)
        # 1 transition  = (départ (list), actions(list), reward(float), arrivée(list), done(bool))
        # conversion in torch tensor
        starts = torch.Tensor([tr[0] for tr in chosentr]).squeeze()
        actions = torch.Tensor([tr[1].detach().numpy() for tr in chosentr]).view(self.mbs, self.action_space_n) 
        rewards = torch.Tensor([tr[2] for tr in chosentr])
        dests = torch.Tensor([tr[3] for tr in chosentr]).squeeze()
        dones = torch.Tensor([tr[4] for tr in chosentr])

        #STEP 12
        # computation of y using Qhat and policyhat of s'/dests
        actionsdestshat = self.policyHat(dests).detach()
        destsactionshat = torch.cat((dests, actionsdestshat), 1)
        qHatValue = self.QvalueHat(
            destsactionshat).squeeze().detach()  # target net
        yHat = rewards + (1 - dones) * (self.gamma * qHatValue).detach()

        self.QvalueOptimizer.zero_grad()
        # compuation of Qphi(s, a) using value network
        # concatenate starts and actions for input of Q net
        startsacts = torch.cat((starts, actions), 1)
        qValue = self.Qvalue(startsacts).squeeze()

        # STEP 13
        valuel = self.QvalueLoss(qValue, yHat)  # compute loss
        valuel.backward()  # gradient step

        self.QvalueOptimizer.step()

        # STEP 14
        self.policyOptimizer.zero_grad()
        actionsstarthat = self.policy(starts)
        startsactionshat = torch.cat((starts, actionsstarthat), 1)
        policyl = - (self.Qvalue(startsactionshat).mean())
        policyl.backward()

        self.policyOptimizer.step()

        # each C steps, update Qhat and policyHat nets
        if self.evts % self.C == 0:
            self.soft_update(self.Qvalue, self.QvalueHat, self.tau)
            self.soft_update(self.policy, self.policyHat, self.tau)

        self.evts += 1

        return self, valuel, - policyl

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward/self.norm_rew, new_ob, done) ########
            self.memory.store(tr)
            # ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.lastTransition = tr

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.opt.freqOptim == 0


if __name__ == '__main__':
    env, config, outdir, logger = init(
        './TP7-8_ContinuousActions/configs/config_pendulum-v0_DDPG.yaml', "DDPGAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DDPG(env, config)

    rsum = 0
    mean = 0
    verbose = False
    itest = 0
    reward = 0
    done = False

    ########################### /!\ dans utils.py ligne 57 changement RL=>RLD ###########################

    for i in range(episode_count):
        # for i in range(1):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  # Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        # Initialize with initial position of the agent
        new_ob = agent.featureExtractor.getFeatures(ob)
        agent.step = 0

        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)  # select action
            new_ob, reward, done, _ = env.step(action.view(agent.action_space_n).detach().numpy())  # execute agent
            new_ob = agent.featureExtractor.getFeatures(
                new_ob)  # extract new observation

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                agent.noise.reset()
                print("forced done!")
            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward

            if agent.timeToLearn(done):
                _, vloss, ploss = agent.learn()

            if done:
                if i % agent.freqVerbose == 0:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) +
                          " actions, value loss=" + str(vloss) + "policy loss=" + str(ploss))
                logger.direct_write("reward", rsum, i)
                logger.direct_write("value loss", vloss, i)
                logger.direct_write("policy loss", ploss, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                agent.noise.reset()

                break

    env.close()
