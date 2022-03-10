"""
    This file contains functions and script to execute the Behavioral clonning approach. 
    For more informations on algorithms see https://dac.lip6.fr/master/rld-2021-2022/.
"""

from expert import toIndexAction, toOneHot
import pickle
from tracemalloc import start
import torch.nn.functional as F
from datetime import datetime
import yaml
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from memory import *
from core import *
from utils import *
import torch
import gym
import argparse
import matplotlib
import copy
matplotlib.use("TkAgg")


class BC(object):
    """
        class that defines the agent using A2C to play
    """

    def __init__(self, env, opt):

        # env and options
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.nbAction = self.action_space.n
        self.featureExtractor = opt.featExtractor(env)
        self.sizeFeature = self.featureExtractor.outSize
        self.freqOptim = opt.freqOptim
        self.freqVerbose = opt.freqVerbose
        self.test = False

        # hyper parameters
        self.nbEvents = 0
        self.evts = 0  # evts

        # Memory
        self.mem_size = opt.mem_size  # mémoire de batches qui contient 10000 transitions
        self.mbs = opt.mbs  # taille des minibatch pour l'apprentissage /!\ mbs <=mem_size
        self.memory = Memory(self.mem_size)

        #Expert
        self.expertActions = None
        self.expertStates = None

        # Actor Critic
        self.Actorlr = opt.actorLr  # actor learning rate
        self.policy = NN(self.featureExtractor.outSize, self.action_space.n, layers=opt.actorLayers, finalActivation=F.softmax, activation=torch.tanh, dropout=0.)  # first NN : learn
        self.ActorOptimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.Actorlr)
        self.lossValue = F.smooth_l1_loss

    # en test on prend le max, en apprentissage on sample selon la politique actuelle
    def act(self, obs):
        """
            Find action by sampling from the probability of the actor
            Input :
                obs : the state of the agent
            Output : 
                The chosen action
        """
        # convert obs to torch object
        tobs = torch.from_numpy(obs).float()  # [1, 8]
        # evaluate Q for current state
        prob = self.policy.forward(tobs)
        # play by smapling according to the distribution from actor
        return np.random.choice(np.arange(0, self.nbAction, 1), p=prob.detach().view(-1).numpy())

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    def state2reward(self, traj, rewards):
        """
            Compute Rewards from the end of the trajectory from the end. 
            Input :
                traj (array) : the trajectory which is a list of states, action, states, rewards, dones
            Output : 
                cumRewards (torch) : The computed cumulated rewards for the considered trajectory
        """
        rewards = rewards.flip(0)
        # print('state2reward', rewards)
        cumRewards = rewards.cumsum(
            0) * 1/(torch.arange(1, rewards.shape[0] + 1, 1))
        # print('state2reward', cumRewards)

        return cumRewards

    # (départ (list), actions(int), reward(float), arrivée(list), done(bool))
    def find1traj(self):
        """
            Extract a complete trajectory from the memory (until a done append)
            Input : 
            Outpit : 
                traj (tensor) : the complete trajectory
        """
        traj = []
        for cpt in range(self.mem_size):
            state = self.memory.getData(cpt)
            traj += [state]
            if state[4] == True:
                break
            if cpt >= self.opt["maxLengthTrain"]:
                break
        return traj

    def loadExpertTransitions(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle)  # .to(agent.floatTensor)
            expert_states = expert_data[:, :self.sizeFeature]
            expert_actions = expert_data[:, self.sizeFeature:]
            expert_states = expert_states.contiguous()
            expert_actions = expert_actions.contiguous()
        self.expertActions = expert_actions
        self.expertStates = expert_states
        return expert_actions, expert_states

    # apprentissage de l'agent
    def learn(self):
        """
            Learning part of the algorithm
            Input : 
            Output : 
                l : the loss of the critic
                gradJ : the loss of the actor
        """

        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return self, 0

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if self.memory.nentities < self.mbs:
            # print('memory not enough filled ', self.memory.nentities, 'available of out ', self.mbs, 'needed for 1 batch')
            return self, 0

        # 1 transition  = (départ (list), actions(int), reward(float), arrivée(list), done(bool))
        # conversion in torch tensor
        # view(-1) sqeeze recrée un tensor alors que view modifie donc view plus rapide
        starts = self.expertStates
        actions = self.expertActions

        log_prob = self.policy(starts)
        loss = - log_prob.sum()
        loss = self.lossValue(log_prob, actions)

        self.ActorOptimizer.zero_grad()
        loss.backward()
        self.ActorOptimizer.step()

        self.evts += 1

        return self, loss

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward, new_ob, done)
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
        './configs/config_BC_lunar.yaml', "BC")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = BC(env, config)
    expert_states, expert_actions = agent.loadExpertTransitions(
        'expert.pkl')  # states are (298, 4), actions are (298, 8)

    rsum = 0
    mean = 0
    verbose = False
    itest = 0
    reward = 0
    done = False

    ########################### /!\ dans utils.py ligne 57 changement RL=>RLD ###########################

    for i in range(episode_count):
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
            new_ob, reward, done, _ = env.step(action)  # execute agent
            new_ob = agent.featureExtractor.getFeatures(
                new_ob)  # extract new observation

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward

            if agent.timeToLearn(done):
                _, loss = agent.learn()

            if done:
                if i % agent.freqVerbose == 0:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) +
                          " actions, loss Discriminatpr =" + str(loss))
                logger.direct_write("reward", rsum, i)
                logger.direct_write("loss", loss, i)

                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
