"""
    This file contains functions and script to execute the A2C algorithm. 
    For more informations on algorithms see https://dac.lip6.fr/master/rld-2021-2022/.
"""

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


class A2C(object):
    """
        class that defines the agent using A2C to play
    """

    def __init__(self, env, opt):

        # env and options
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile) # TODO : add opt files 
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
        self.explo = 0.1  # parameter for balancing exploration in epsilon greedy
        self.decay = 0.9  # parameter for decreasing the exploration when learning
        self.gamma = 0.99  # discount
        self.nbTr = 1  # number of transition per batch for learning
        self.C = 10  # update target each C steps

        # Memory
        self.mem_size = 1000  # mémoire de batches qui contient 10000 transitions
        self.mbs = 1000  # taille des minibatch pour l'apprentissage /!\ mbs <=mem_size
        self.memory = Memory(self.mem_size)

        # Actor Critic
        self.Actorlr = 0.001  # actor learning rate
        self.Valuelr = 0.001  # critic learning rate
        self.actor = NN(self.featureExtractor.outSize, self.action_space.n, layers=[
                        30, 30], finalActivation=F.softmax, activation=torch.tanh, dropout=0.)  # first NN : learn
        self.value = NN(self.featureExtractor.outSize, 1, layers=[
                        30, 30], finalActivation=None, activation=torch.tanh, dropout=0.)  # second NN : play
        self.ActorOptimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.Actorlr)
        self.ValueOptimizer = torch.optim.Adam(
            self.value.parameters(), lr=self.Valuelr)
        self.lossValue = F.smooth_l1_loss

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
        tobs = torch.from_numpy(obs).float()
        # evaluate Q for current state
        prob = self.actor.forward(tobs)
        # play by smapling according to the distribution from actor
        return np.random.choice(np.arange(0, self.nbAction, 1), p=prob.detach().view(-1).numpy())

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    def state2reward(self, traj):
        """
            Compute Rewards from the end of the trajectory from the end. 
            Input :
                traj (array) : the trajectory which is a list of states, action, states, rewards, dones
            Output : 
                cumRewards (torch) : The computed cumulated rewards for the considered trajectory
        """
        starts = torch.Tensor([tr[0] for tr in traj]).squeeze().flip(0)
        rewards = torch.Tensor([tr[2] for tr in traj])
        cumRewards = torch.zeros(rewards.size())
        cumRewards[-1] = rewards[-1]
        for j in np.arange(rewards.size()[0] - 1, 1, -1):
            cumRewards[j-1] = rewards[j-1] + self.gamma * cumRewards[j]
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
            return self, 0, 0

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if self.memory.nentities < self.mbs:
            # print('memory not enough filled ', self.memory.nentities, 'available of out ', self.mbs, 'needed for 1 batch')
            return self, 0, 0

        # Sinon on calcule y et on fait une descente de gradient
        # Extract trajectories
        chosentr = self.find1traj()
        # 1 transition  = (départ (list), actions(int), reward(float), arrivée(list), done(bool))
        # conversion in torch tensor
        # view(-1) sqeeze recrée un tensor alors que view modifie donc view plus rapide
        starts = torch.Tensor([tr[0] for tr in chosentr]).squeeze()
        actions = torch.Tensor([tr[1] for tr in chosentr]).type(
            torch.int64)  # actions are int because they are discrete
        rewards = torch.Tensor([tr[2] for tr in chosentr])
        dests = torch.Tensor([tr[3] for tr in chosentr]).squeeze()
        dones = torch.Tensor([tr[4] for tr in chosentr])
        cumRewards = self.state2reward(chosentr)

        # Update Value/Critic network
        yHat = self.value(starts).view(-1)
        l = self.lossValue(yHat, cumRewards)
        l.backward()
        self.ValueOptimizer.step()
        self.ValueOptimizer.zero_grad()

        # Evaluate advantage
        with torch.no_grad():
            achap = rewards + self.gamma * \
                self.value(dests).view(-1) - self.value(starts).view(-1)

        # Compute Actor gradient
        piTeta = self.actor(starts).view(-1, self.nbAction)
        gradJ = -(torch.log(piTeta.gather(1, actions.unsqueeze(1))) * achap).sum()
        # Update Policy/Actor
        gradJ.backward()
        # Optimize the policy and reset the memory ==> if we change the policy, the
        self.ActorOptimizer.step()
        self.ActorOptimizer.zero_grad()
        # reset memory
        del self.memory
        self.memory = Memory(self.mem_size)

        # # each C steps, update Qhat net
        # if self.evts % self.C == 0:
        #     self.Qhat.load_state_dict(self.Q.state_dict())
        self.evts += 1

        return self, l, gradJ

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
        './configs/config_A2C_lunar.yaml', "A2CAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = A2C(env, config)

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
                _, lossValue, lossActor = agent.learn()

            if done:
                if i % agent.freqVerbose == 0:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) +
                          " actions, loss Value =" + str(lossValue) + "loss Value =" + str(lossActor))
                logger.direct_write("reward", rsum, i)
                logger.direct_write("loss for Value Network", lossValue, i)
                logger.direct_write("loss for Policy Network", lossActor, i)

                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
