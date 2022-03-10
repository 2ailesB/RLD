"""
    This file contains functions and script to execute the GAIL algorithm. 
    For more informations on algorithms see https://dac.lip6.fr/master/rld-2021-2022/.
"""

from expert import toIndexAction, toOneHot
import pickle
from torch.distributions import Categorical, Normal
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


class GAIL(object):
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
        self.noptim = 0
        self.evts = 0  # evts
        self.gamma = opt.gamma  # discount
        self.nbTr = opt.nbTr  # number of transition per batch for learning
        self.C = opt.C  # update target each C steps
        self.K = opt.K  # nb updates
        self.beta = 1
        self.delta = opt.delta
        self.epsilon = opt.epsilon
        self.ew = opt.entWeight
        self.rewards = opt.rewards  # 'pos' or 'neg' or None
        self.eps_d = opt.eps_d

        # Memory
        self.mem_size = opt.mem_size  # mémoire de batches qui contient 10000 transitions
        self.mbs = opt.mbs  # taille des minibatch pour l'apprentissage /!\ mbs <=mem_size
        self.memory = Memory(self.mem_size)
        self.gailmemory = Memory(self.mem_size)

        #Expert
        self.expertActions = None
        self.expertStates = None

        # Actor Critic
        self.Actorlr = opt.ActorLr  # actor learning rate
        self.Valuelr = opt.ValueLr  # critic learning rate
        self.Discriminatorlr = opt.DiscLr  # critic learning rate
        self.policy = NN(self.featureExtractor.outSize, self.action_space.n, layers=opt.ActorLayers, finalActivation=torch.nn.Softmax(-1), activation=torch.tanh, dropout=0.)  # first NN : learn
        self.value = NN(self.featureExtractor.outSize, 1, layers=opt.ValueLayers, finalActivation=None, activation=torch.tanh, dropout=0.)  # second NN : play
        self.discriminator = NN(self.featureExtractor.outSize + self.nbAction, 1, layers=opt.DiscLayers, finalActivation=torch.nn.Sigmoid(), activation=torch.tanh, dropout=0.)
        self.discriminatorLoss = nn.BCEWithLogitsLoss()
        # self.discriminatorLoss = nn.BCELoss()
        self.ActorOptimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.Actorlr)
        self.ValueOptimizer = torch.optim.Adam(
            self.value.parameters(), lr=self.Valuelr)
        self.DiscriminatorOptimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.Discriminatorlr)
        self.lossValue = F.smooth_l1_loss
        self.algo = opt.algo  # or 'KL' or 'clipped or 'A2C'

        self.sigma = opt.sigma
        self.noise = Normal(0, self.sigma)

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
        tobs = torch.from_numpy(obs).float()  # [1, 8]
        # evaluate Q for current state
        prob = self.policy.forward(tobs) # [1, 4]
        # # play by smapling according to the distribution from actor
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
        cumRewards = rewards.cumsum(
            0) * 1/(torch.arange(1, rewards.shape[0] + 1, 1))

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
            return self, 0, 0, 0, 0

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if self.memory.nentities < self.mbs:
            # print('memory not enough filled ', self.memory.nentities, 'available of out ', self.mbs, 'needed for 1 batch')
            return self, 0, 0, 0, 0

        # Sample Agent
        z, _, chosentr = self.memory.sample(self.mbs)
        starts = torch.Tensor([tr[0] for tr in chosentr]).view(
            len(chosentr), self.sizeFeature)  # (70, 8)
        actions = torch.Tensor([tr[1] for tr in chosentr]).type(
            torch.int64)  # actions are int because they are discrete (70) one hot version s [70, 4]
        # rewards = torch.Tensor([tr[2] for tr in chosentr])
        dests = torch.Tensor([tr[3] for tr in chosentr]).squeeze()
        dones = torch.Tensor([tr[4] for tr in chosentr])

        # sample expert actions
        idx = np.random.choice(range(len(self.expertStates)), len(chosentr))
        expert_starts_actions = torch.cat((self.expertStates, self.expertActions), dim=1)[
            idx, :]  # expert starts are [298, 8] and actions are [298, 4]
            
        # discriminator updates
        # compute disc for expert
        noise = self.noise.sample(expert_starts_actions.shape)
        expertsteps = self.discriminator(
            expert_starts_actions + noise).view(expert_starts_actions.shape[0])
        expert_l = self.discriminatorLoss(
            expertsteps, torch.ones(expertsteps.shape[0]))
        # expert_l = torch.log(expertsteps)
        # compute disc for agent
        starts_actions = torch.cat(
            (starts, toOneHot(self.env, actions.unsqueeze(1))), dim=1)
        noise = self.noise.sample(starts_actions.shape)
        modelsteps = self.discriminator(
            starts_actions + noise).view(starts_actions.shape[0])  # [70, 1]
        model_l = self.discriminatorLoss(
            modelsteps, torch.zeros(actions.shape[0]))
        # model_l = torch.log(torch.ones_like(modelsteps) - modelsteps)
        disc_l = expert_l + model_l
        # disc_l = - (expert_l + model_l).mean()
        self.DiscriminatorOptimizer.zero_grad()
        disc_l.backward()
        self.DiscriminatorOptimizer.step()

        # compute new rewards based on new generator
        _, rewards = self.compute_r()
        rewards = [rewards[i].detach() for i in z]
        rewards = torch.Tensor(rewards)

        # Evaluate advantage
        with torch.no_grad():
            achap = rewards.detach() -  self.value(starts).detach() 
            # achap = rewards.detach() -  self.gamma * self.value(dests).view(-1) - self.value(starts).view(-1)                    

        # Compute Actor gradient
        gradJ_Theta = 0

        if self.algo == 'clipped':
            piTheta_k = self.policy(starts).view(-1, self.nbAction).detach().gather(
                1, actions.unsqueeze(1))  # not changed during loop
            for i in range(self.K):
                piTheta1 = self.policy(starts).view(-1, self.nbAction)
                piTheta = piTheta1.gather(1, actions.unsqueeze(1))
                ratios = (piTheta) / (piTheta_k)
                gradJ_Theta = ratios * achap.detach()
                lossP = (torch.clamp(
                    ratios, 1 - self.epsilon, 1 + self.epsilon) * achap)
                lossP = - torch.min(gradJ_Theta, lossP).mean()
                # entropy = - (piTheta * torch.log(piTheta)).mean()
                entropy = Categorical(piTheta1).entropy().mean()
                lossP = (lossP - self.ew * entropy)
                self.ActorOptimizer.zero_grad()
                lossP.backward()
                self.ActorOptimizer.step()

        elif self.algo == 'KL':
            # not changed during loop
            piTheta_k = self.policy(starts).view(-1, self.nbAction).detach()
            for i in range(self.K):
                self.ActorOptimizer.zero_grad()
                piTheta = self.policy(starts).view(-1, self.nbAction)
                gradJ_Theta = ((piTheta.gather(1, actions.unsqueeze(
                    1)) - (piTheta_k).gather(1, actions.unsqueeze(1))) * achap).mean()
                gradJ_Theta -= self.beta * \
                    (F.kl_div(piTheta_k, piTheta)).mean()
                lossP = - gradJ_Theta  # gradient ascent
                # Update policy
                lossP.backward()
                self.ActorOptimizer.step()
            piTheta_kp1 = self.policy(starts).view(-1, self.nbAction).detach()
            if self.beta * (F.kl_div(piTheta_k, piTheta_kp1)).mean() > 1.5 * self.delta:
                self.beta = 2 * self.beta
            if self.beta * (F.kl_div(piTheta_k, piTheta_kp1)).mean() < self.delta / 1.5:
                self.beta = 0.5 * self.beta

        elif self.algo == 'A2C':
            # Compute Actor gradient
            piTeta = self.policy(starts).view(-1, self.nbAction)
            lossP = -(torch.log(piTeta.gather(1, actions.unsqueeze(1))) * achap).sum()
            # Update Policy/Actor
            lossP.backward()
            # Optimize the policy and reset the memory ==> if we change the policy, the
            self.ActorOptimizer.step()
            self.ActorOptimizer.zero_grad()

        # Update Value/Critic network

        yHat = self.value(starts).view(-1)
        l = self.lossValue(yHat, rewards.detach())
        self.ValueOptimizer.zero_grad()
        l.backward()
        self.ValueOptimizer.step()

        # reset memory
        del self.memory
        self.memory = Memory(self.mem_size)
        del self.gailmemory
        self.gailmemory = Memory(self.mbs)

        self.evts += 1

        return self, expertsteps.mean(), modelsteps.mean(), disc_l, lossP, l

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage,
            # alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward, new_ob, done)
            self.memory.store(tr)
            # ici on n'enregistre que la derniere transition pour traitement immédiat,
            # mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.lastTransition = tr

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self, done):
        if self.test:
            return False
        if self.memory.nentities < self.mbs:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.opt.freqOptim == 0

    def compute_r(self):
        cumulated_rewards = []
        cumulated_rewards_real = []
        r_cumulated = 0
        r_cumulated_real = 0
        remaining_ep_len = 0

        starts = torch.Tensor([self.memory.getData(i)[0] for i in range(
            self.memory.nentities)]).view(self.memory.nentities, self.sizeFeature)
        actions = torch.Tensor([self.memory.getData(i)[1] for i in range(
            self.memory.nentities)]).type(torch.int64)
        actions = toOneHot(self.env, actions)

        # get GAIL rewards (using the discriminator) instead of real rewards
        startsactions = torch.cat([starts, actions], dim=1)  # 1000, 8
        d_agent = self.discriminator(startsactions)  # 1000, 4
        if self.rewards == 'neg':
            d_agent = torch.clamp(1 - d_agent, self.eps_d, 1 - self.eps_d)
            rewards_clipped = torch.clamp(torch.log(d_agent), min=-100, max=0)
        elif self.rewards == 'pos':
            d_agent = torch.clamp(1 - d_agent, self.eps_d, 1 - self.eps_d)
            rewards_clipped = torch.clamp(- torch.log(1 -
                                          d_agent), min=-100, max=0)
        else:
            rewards_clipped = d_agent

        for i in reversed(range(self.memory.nentities)):
            s, a, r, s_prime, done = self.memory.getData(i)
            if done:
                remaining_ep_len = 0
                r_cumulated = 0
                r_cumulated_real = 0
            r_cumulated_real = r + r_cumulated_real
            r_cumulated = rewards_clipped[i] + r_cumulated
            remaining_ep_len += 1
            cumulated_rewards.append(r_cumulated / remaining_ep_len)
            cumulated_rewards_real.append(r_cumulated_real / remaining_ep_len)
            self.gailmemory.store(
                (s, a, (r_cumulated / remaining_ep_len).detach().item(), s_prime, done))
        return self, list(reversed(cumulated_rewards))


if __name__ == '__main__':
    env, config, outdir, logger = init(
        './configs/config_GAIL_lunar.yaml', "GAIL")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = GAIL(env, config)
    expert = agent.loadExpertTransitions('expert.pkl')

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
                _, expertDiscProb, modelDiscProb, lossDisc, lossActor, lossValue = agent.learn()
                logger.direct_write(
                    "Prob for disciminator on Model", modelDiscProb, i)
                logger.direct_write(
                    "Prob for disciminator on Expert", expertDiscProb, i)
                logger.direct_write("loss for disciminator", lossDisc, i)
                logger.direct_write("loss for Policy Network", lossActor, i)
                logger.direct_write("loss for Value Network", lossValue, i)

            if done:
                if i % agent.freqVerbose == 0:
                    print(str(i) + " rsum=" + str(rsum) +
                          ", " + str(j) + " actions")
                logger.direct_write("reward", rsum, i)

                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
