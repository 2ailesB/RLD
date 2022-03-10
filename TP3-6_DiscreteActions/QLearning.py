from torch.utils.tensorboard import SummaryWriter
from utils import *
import gridworld
from gym import logger, wrappers
import numpy as np
import gym
from datetime import datetime
import os
import copy
import matplotlib

matplotlib.use("TkAgg")


class QLearning(object):

    def __init__(self, env, opt):

        self.opt = opt
        self.action_space = env.action_space
        self.env = env
        self.discount = opt.gamma
        self.alpha = opt.learningRate
        self.explo = opt.explo
        self.exploMode = opt.exploMode  # 0: epsilon greedy, 1: ucb
        self.sarsa = opt.sarsa
        self.modelSamples = opt.nbModelSamples
        self.test = False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles

    def save(self, file):
       pass

    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self, obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0)
            # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
        return ss

    def epsilonGreedy(self, muchap, T, epsilon):
        eps = np.random.uniform()
        if eps < epsilon:
            return np.random.randint(0, len(muchap))
        return np.argmax(muchap)

    # en test on prend le max, en apprentissage on fait un epsilon greddy pour explorer au hasard de temps en temps
    def act(self, obs):
        if self.test:
            return np.argmax(self.values[obs])
        else:
            self.explo *= self.discount
            return self.epsilonGreedy(self.values[obs], 1, self.explo)

    def store(self, ob, action, new_ob, reward, done, it):

        if self.test:
            return
        self.last_source = ob
        self.last_action = action
        self.last_dest = new_ob
        self.last_reward = reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done = done

    def learn(self, done):
        self.values[self.last_source][self.last_action] += self.alpha * (self.last_reward + self.discount * (
            np.max(self.values[self.last_dest] - self.values[self.last_source][self.last_action])))


if __name__ == '__main__':
    env, config, outdir, logger = init(
        './TP3-6_DiscreteActions/configs/config_gridworld_qlearning.yaml', "QLearning")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = QLearning(env, config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0
    for i in range(episode_count):
        # permet de changer la config en cours de run
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        if (i > 0 and i % int(config["freqVerbose"]) == 0):
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  # Si agent.test alors retirer l'exploration
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.storeState(ob)

        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)  # choose action
            new_ob, reward, done, _ = env.step(action)  # apply action
            new_ob = agent.storeState(new_ob)

            j += 1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                #print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            agent.learn(done)
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) +
                      ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                break

    env.close()
