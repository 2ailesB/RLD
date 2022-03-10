"""
    This file contains functions and script to execute the goalSampling techniques based on DQN algorithm. 
    For more informations on algorithms see https://dac.lip6.fr/master/rld-2021-2022/.  
"""


from collections import deque
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
matplotlib.use("TkAgg")


class goalSampling(object):
    """
        class that defines the agent using DQN to play
    """

    def __init__(self, env, opt):

        # parameters of the environnement
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test = False
        self.nbEvents = 0
        self.evts = 0  # number of call of the learn function

        # parameters of the algorithm
        self.explo = opt.explo  # parameter for balancing exploration in epsilon greedy
        self.decay = opt.decay  # parameter for decreasing the exploration when learning
        self.gamma = opt.gamma  # parameter for discount
        self.lr = opt.lr  # learning rate
        self.target = opt.target  # wether we use target network or not
        self.replay = opt.replay  # wether we use replay buffer or not
        self.prior = opt.prior  # wether we use the prioritized version or not
        self.freqOptimQtarget = opt.C  # update target each C steps

        # parameters for goal sampling
        self.beta = opt.beta  # Probability to sample a fake goal
        self.betadecay = opt.betadecay  # decrease the probability to sample fake goals
        self.nGoals = opt.nGoals
        self.goals = deque(maxlen=self.nGoals)
        self.ng = deque(maxlen=self.nGoals)
        self.vg = deque(maxlen=self.nGoals)
        self.alpha = opt.alpha
        self.fake_goals = False
        self.freqAddG = opt.freqAddG

        # Memory
        self.mem_size = opt.mem_size  # mémoire de batches qui contient 10000 transitions
        self.mbs = opt.mbs  # taille des minibatch pour l'apprentissage /!\ mbs <=mem_size
        self.memory = Memory(self.mem_size, prior=self.prior)  # Memory

        self.mem_size2 = opt.mem_size2
        self.episodeMemory = Memory(self.mem_size2, prior=self.prior)

        # Networks
        self.Q = NN(2 * self.featureExtractor.outSize, self.action_space.n, layers=opt.QLayers, finalActivation=None, activation=torch.tanh)  # first NN : learn
        self.Qhat = copy.deepcopy(self.Q)  # second NN : play
        self.Qmean = 0
        if not self.target:
            # si on n'utilise pas la target, Q = Qtarget, ie si on update Q on update aussi Qtarget.
            self.freqOptimQtarget = 1
        self.freqOptim = opt.freqOptim
        self.freqVerbose = opt.freqVerbose
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.loss = F.smooth_l1_loss

    def epsilonGreedy(self, muchap, epsilon):
        """
            Epsilon-Greedy exploration
            Input :
                muchap (array) : array on which we choose action
                epsilon (float) : threshold for random choice
            Output : 
                The indice of the chosen action
        """

        eps = np.random.uniform()
        if eps < epsilon:
            return self.action_space.sample()
        return torch.argmax(muchap).item()

    # en test on prend le max, en apprentissage on fait un epsilon greddy pour explorer au hasard de temps en temps
    def act(self, obs, goal):
        """
            Find action with epsilon greedy strategy in train and max in test
            Input :
                obs : the state of the agent
            Output : 
                The chosen action
        """
        #convert obs to torch object
        tobs = torch.from_numpy(obs).float()  # [1, 2]
        # [1, 2] => cat(, dim=1) donne [1, 4]
        goal = torch.from_numpy(goal).float()
        # evaluate Q for current state
        Qval = self.Q.forward(torch.cat((tobs, goal), dim=1))
        if self.test:
            return torch.argmax(Qval).item()
        else:
            self.explo *= self.decay
            return self.epsilonGreedy(Qval, self.explo)

    # apprentissage de l'agent
    def learn(self):
        """
            Learning part of the algorithm
            Input : 
            Output : 
                l : the loss computed
        """

        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return self, 0

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if self.memory.nentities <= self.mbs:
            print('memory not enough filled ', self.memory.nentities,
                  'available of out ', self.mbs, 'needed for 1 batch')
            return self, 0

        # sinon on calcule y et on fait une descente de gradient
        # On tire aléatoirement un mini batch de transitions
        chosenIdx, z, chosentr = self.memory.sample(self.mbs)
        # 1 transition  = (départ (list), actions(int), reward(float), arrivée(list), done(bool))
        # conversion in torch tensor
        starts = torch.Tensor([tr[0] for tr in chosentr]).squeeze()  # [100, 2]
        actions = torch.Tensor([tr[1] for tr in chosentr]).type(torch.int64)
        rewards = torch.Tensor([tr[2] for tr in chosentr])  # [100]
        dests = torch.Tensor([tr[3] for tr in chosentr]).squeeze()  # [100, 2]
        dones = torch.Tensor([tr[4] for tr in chosentr])
        goals = torch.Tensor([tr[5] for tr in chosentr]).squeeze()  # [100, 2]

        # Calcul des quantités
        startsgoals = torch.cat((starts, goals), dim=1)
        destsgoals = torch.cat((dests, goals), dim=1)
        qValue = (self.Q(startsgoals).gather(1, actions.view(-1, 1))
                  ).squeeze()  # learning net
        self.Qmean = qValue.mean().detach()
        qHatValue = self.Qhat(destsgoals).detach()  # target net
        # take the max value per batch and compute the prediction max(1) to have one per transition and [0] to consider the value and not the index
        yHat = rewards + (1 - dones) * (self.gamma *
                                        qHatValue.max(1)[0].detach())

        # Compute loss and gradients
        l = self.loss(yHat, qValue)  # compute loss
        l.backward()  # gradient step

        # Optimizaiton step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Optimization of target net
        if self.evts % self.freqOptimQtarget == 0:
            self.Qhat.load_state_dict(self.Q.state_dict())
        self.evts += 1

        return self, l

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, goal, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward, new_ob, done, goal)
            self.memory.store(tr)
            # ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.lastTransition = tr

    # enregistrement de la transition pour exploitation par learn ulterieure
    def storeEpisode(self, ob, action, new_ob, reward, done, goal, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward, new_ob, done, goal)
            self.episodeMemory.store(tr)
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

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    def addGoal(self, lentraj):
        # get last state meet ie the new goal
        new_goal = self.episodeMemory.getData(lentraj-1)[3]
        # self.memory.nentities gives the number of stored elements
        for i in range(self.episodeMemory.nentities):
            ob, action, reward, new_ob, done, goal = self.episodeMemory.getData(
                i)
            if (new_goal == self.episodeMemory.getData(i)[3]).all():
                reward = 1
                done = True
            self.store(ob, action, new_ob, reward, done,
                       new_goal, i + self.memory.nentities)

    def addGoaledTr(self):
        for i in range(self.episodeMemory.nentities):
            tr = self.episodeMemory.getData(i)
            self.memory.store(tr[0], tr[1], tr[2], tr[3], tr[4], tr[5])

    def epMemReset(self):
        self.episodeMemory = Memory(self.mem_size2, prior=self.prior)

    def AgentSampleGoal(self, env):
        self.beta *= self.betadecay
        # succes with proba self.beta => fake goals
        b = np.random.binomial(n=1, p=self.beta, size=1)
        if 1-b or len(self.goals) == 0:
            g, _ = env.sampleGoal()
            g = self.featureExtractor.getFeatures(g)
            self.fake_goals = False
            idx = None
            print('true goal sampled')
        else:
            quot = np.array(self.dequeDivide(self.vg, self.ng))
            Hg = -quot * np.log(quot) - (1 - quot) * np.log(1 - quot)
            prob = np.exp(self.alpha * Hg)
            idx = np.random.choice(np.arange(0, len(
                self.goals)), size=1, replace=True, p=np.exp(prob)/(np.exp(prob).sum()))[0]
            g = self.goals[idx]
            self.fake_goals = True
            print('fake goal sampled')
        return g, idx

    def newGoal(self, ob):
        if len(self.goals) == self.goals.maxlen:
            # décale les éléments vers la gauche ie premier elt à la fin = le plus vieux
            self.goals.rotate(-1)
            self.goals.pop()  # on retire le dernier elt = le plus vieux
            # Donc on ajoute le nouveau goal à la fin = à la place du premier goal
            self.goals.insert(len(self.goals), ob)
            self.ng.rotate(-1)
            self.ng.pop()
            self.ng.insert(len(self.goals), 1 - 1e-10)
            self.vg.rotate(-1)
            self.vg.pop()
            self.vg.insert(len(self.goals), 0 + 1e-10)
        else:
            self.goals.append(ob)  # sinon on complète la deque
            self.ng.append(1 - 1e-10)
            self.vg.append(0 + 1e-10)

    def dequeDivide(self, list1, list2):
        return([a/b for a, b in zip(list1, list2)])

    def dequeExist(self, list1, obs):
        exist = False
        for elt in list1:
            if (elt == obs).all():
                exist = True
        return exist


if __name__ == '__main__':
    env, config, outdir, logger = init(
        './configs/config_gridworld_GoalSampling.yaml', "goalSamplingAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = goalSampling(env, config)

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
        goal, goal_idx = agent.AgentSampleGoal(env)
    
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
            
            action = agent.act(ob, goal)
            new_ob, _, _, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)
            done = (new_ob == goal).all()
            reward = 1.0 if done else -0.1
            forced_done = False

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                forced_done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done, goal, j)
            agent.storeEpisode(ob, action, new_ob, reward, done, goal, j)
            rsum += reward

            if agent.timeToLearn(done):
                _, loss = agent.learn()

            if done:
                if i % agent.freqVerbose == 0:
                    print(str(i) + " rsum=" + str(rsum) + ", " +
                          str(j) + " actions, loss=" + str(loss))
                logger.direct_write("reward", rsum, i)
                logger.direct_write("loss", loss, i)
                logger.direct_write("Q moyen", agent.Qmean, i)
                logger.direct_write('xfinish', new_ob[0][0], i)
                logger.direct_write('yfinish', new_ob[0][1], i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                if not agent.test:
                    # if episode has ended then we add goals and save them in the principal memory
                    agent.addGoal(j)
                    # agent.addGoaledTr() # add the modified transition to buffer
                    agent.epMemReset()

                if agent.fake_goals:
                    agent.ng[goal_idx] += 1
                    if not forced_done:
                        agent.vg[goal_idx] += 1
                break

        if i % agent.freqAddG == 0:
            #add a new goal
            if not agent.dequeExist(agent.goals, ob):
                agent.newGoal(ob)

    env.close()
