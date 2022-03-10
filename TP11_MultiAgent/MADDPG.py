import random
from datetime import datetime
import yaml
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from memory import *
from core import *
from utils import *
from DDPG import DDPG
import torch
import gym
import argparse
import sys
import matplotlib
import copy
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")


def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    nact = world.dim_c
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    world.dim_c = nact
    return env, scenario, world


class MADDPG(object):
    """
        class that defines the agent using DDPG to play
    """

    def __init__(self, env, scenario, world, opt):
        o = env.reset()
        self.opt = load_yaml(opt)
        obs_n = [os.shape[0] for os in o]
        mshape = max(obs_n)
        o = padObs(o, mshape)

        self.nAgents = len(o)
        self.obsSize = len(o[0])
        self.nActions = world.dim_c
        self.mbs = self.opt.mbs

        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
        self.outdir = "./XP/" + self.opt.env + "/" + "MADDPG" + "_" + date_time

        self.test = False

        self.evts = 0  # evts
        self.freqOptim = self.opt.freqOptim
        self.freqVerbose = self.opt.freqVerbose

        self.agents = [DDPG(env, world, scenario, self.opt)
                       for i in range(self.nAgents)]

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
    def act(self, obs):
        """
            Find action by sampling from the probability of the actor + a chosen noise
            Input :
                obs : the state of the agent
            Output : 
                pol : The actions to be played
        """

        pol = []
        for idx, ag in enumerate(self.agents):
            tobs = torch.from_numpy(obs[idx]).float()
            epsilon = ag.noise.sample()

            polAg = ag.policyHat.forward((tobs).float())
            if not self.test:  # En test on n'explore plus donc on n'ajoute pas le espilon
                polAg += epsilon
            pol += [polAg.clamp(-1, 1)]

        pol = torch.cat(pol).view(self.nAgents, self.nActions)
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
            return self, self.nAgents*[0], self.nAgents*[0]

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if self.agents[0].memory.nentities <= self.mbs:
            print('memory not enough filled ', self.agents[0].memory.nentities,
                  'available of out ', self.mbs, 'needed for 1 batch')
            return self, self.nAgents*[0], self.nAgents*[0]

        v_losses = []
        p_losses = []

        for idx_ag, ag in enumerate(self.agents):
            _, vloss, ploss = self.learn1agent(ag, idx_ag)
            v_losses.append(vloss)
            p_losses.append(ploss)

        return self, v_losses, p_losses

    def learn1agent(self, cur_agent, idx_agent):
        """
            Learning part of the algorithm
            Input : 
            Output : 
                valuel : the loss of the Value function
                -policyl : the loss of the policy
        """
        # Si l'agent est en mode de test, on n'entraîne pas
        if cur_agent.test:
            return cur_agent, 0, 0

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if cur_agent.memory.nentities <= cur_agent.mbs:
            print('memory not enough filled ', cur_agent.memory.nentities,
                  'available of out ', cur_agent.mbs, 'needed for 1 batch')
            return cur_agent, 0, 0

        # sinon on calcule y et on fait une descente de gradient
        # On tire aléatoirement un mini batch de transitions
        _, _, chosentr = cur_agent.memory.sample(cur_agent.mbs)
        # 1 transition  = (départ (list), actions(list), reward(float), arrivée(list), done(bool))
        # conversion in torch tensor
        starts, actions, rewards, dests, dones, starts_i, actions_i, rewards_i, dests_i, dones_i = self.prepareBatch(
            chosentr, cur_agent, idx_agent)
        #STEP 12
        # computation of y using Qhat and policyhat of s'/dests
        actionshat = ([ag.policyHat(dests_i[:, idx, :]).detach()
                      for idx, ag in enumerate(self.agents)])
        actionshat = torch.cat(actionshat, 1)  # 100, 6
        destsactionshat = torch.cat((dests, actionshat), 1)  # 100, 60

        qHatValue = cur_agent.QvalueHat(
            destsactionshat).squeeze().detach()  # target net
        yHat = rewards_i[:, idx_agent] + \
            (1 - dones_i[:, idx_agent]) * \
            (cur_agent.gamma * qHatValue)  # (&00)
        # compuation of Qphi(s, a) using value network
        # concatenate starts and actions for input of Q net
        startsacts = torch.cat((starts, actions), 1)  # (100, 60)
        qValue = cur_agent.Qvalue(startsacts).squeeze()  # (100)

        # STEP 13
        valuel = cur_agent.QvalueLoss(qValue, yHat.detach())  # compute loss
        cur_agent.QvalueOptimizer.zero_grad()
        valuel.backward()  # gradient step

        cur_agent.QvalueOptimizer.step()

        # STEP 14
        actionshat = ([ag.policy(dests_i[:, idx, :])
                      for idx, ag in enumerate(self.agents)])  # policy ?
        actionshat = torch.cat(actionshat, 1)  # 100, 6
        startsactionshat = torch.cat((starts, actionshat), 1)  # (100, 60)

        policyl = - (cur_agent.Qvalue(startsactionshat).mean(dim=0))
        cur_agent.policyOptimizer.zero_grad()
        policyl.backward()
        cur_agent.policyOptimizer.step()

        # each C steps, update Qhat and policyHat nets
        if cur_agent.evts % cur_agent.C == 0:
            with torch.no_grad():
                cur_agent.soft_update(
                    cur_agent.Qvalue, cur_agent.QvalueHat, cur_agent.tau)
                cur_agent.soft_update(
                    cur_agent.policy, cur_agent.policyHat, cur_agent.tau)

        cur_agent.evts += 1

        return cur_agent, valuel, - policyl

    def prepareBatch(self, batch, cur_agent, i):
        """
        convert list to tensor and extract the actions and state of the current agent
        """

        # all info from all agents
        starts = torch.Tensor([tr[0] for tr in batch]).squeeze().view(
            cur_agent.mbs, self.nAgents * self.obsSize)  # (100, 3, 18) => (100, 54)
        actions = torch.Tensor([tr[1].detach().numpy() for tr in batch]).view(
            cur_agent.mbs, self.nAgents*self.nActions)  # [100, 3, 2] ==> (100, 6)
        rewards = torch.Tensor([tr[2] for tr in batch]).view(
            cur_agent.mbs, self.nAgents * 1)  # [100, 3]
        dests = torch.Tensor([tr[3] for tr in batch]).squeeze().view(
            cur_agent.mbs, self.nAgents * self.obsSize)  # (100, 3, 18) ==> (100, 54)
        dones = torch.Tensor([tr[4] for tr in batch]).view(
            cur_agent.mbs, self.nAgents * 1)  # [100, 3]
        # torch.Size([100, 54]) torch.Size([100, 6]) torch.Size([100, 3]) torch.Size([100, 54]) torch.Size([100, 3])

        # info from current agent
        starts_i = torch.Tensor([tr[0] for tr in batch]).squeeze()
        actions_i = torch.Tensor([tr[1].detach().numpy() for tr in batch])
        rewards_i = torch.Tensor([tr[2] for tr in batch])
        dests_i = torch.Tensor([tr[3] for tr in batch]).squeeze()
        dones_i = torch.Tensor([tr[4] for tr in batch])
        # with [i] torch.Size([100, 18]) torch.Size([100, 2]) torch.Size([100]) torch.Size([100, 18]) torch.Size([100])
        # torch.Size([100, 3, 18]) torch.Size([100, 3, 2]) torch.Size([100, 3]) torch.Size([100, 3, 18]) torch.Size([100, 3])

        return starts, actions, rewards, dests, dones, starts_i, actions_i, rewards_i, dests_i, dones_i

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = agent.nAgents*[False]
            for idx, ag in enumerate(self.agents):
                tr = (ob, action, reward, new_ob, done)

                ag.memory.store(tr)
                # ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
                ag.lastTransition = tr

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.opt.freqOptim == 0

# help functions
def add_list(list1, list2):
    return [x + y for (x, y) in zip(list1, list2)]


def mul_list(list1, list2):
    return [x * y for (x, y) in zip(list1, list2)]


def add_list_scalar(list1, scalar):
    return [x + scalar for x in list1]


def mul_list_scalar(list1, scalar):
    return [x * scalar for x in list1]


def list_clip(list1, scalarmin, scalarmax):
    return [max(min(x, scalarmax), scalarmin) for x in list1]


def padObs(obs, size):
    return([np.concatenate((o, np.zeros(size-o.shape[0]))) if o.shape[0] < size else o for o in obs])


if __name__ == '__main__':

    env, scenario, world = make_env('simple_spread')

    agent = MADDPG(env, scenario, world, opt='./configs/simple_spread.yaml')
    logger = LogMe(SummaryWriter(agent.outdir))
    loadTensorBoard(agent.outdir)

    config = load_yaml('./configs/simple_spread.yaml')

    o = env.reset()
    obs_n = [os.shape[0] for os in o]
    mshape = max(obs_n)
    o = padObs(o, mshape)
    obs_n = [mshape for os in o]
    reward = agent.nAgents*[0]
    rsum = agent.nAgents*[0]

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    rsum = agent.nAgents*[0]
    mean = agent.nAgents*[0]
    verbose = False
    itest = 0
    reward = agent.nAgents*[0]
    done = agent.nAgents*[False]

    ########################### /!\ dans utils.py ligne 57 changement RL=>RLD ###########################

    for i in range(episode_count):
        checkConfUpdate(agent.outdir, config)

        rsum = agent.nAgents*[0]
        agent.nbEvents = 0
        ob = env.reset()
        ob = padObs(ob, mshape)

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  # Same as train for now
            print("Test time! ")
            mean = agent.nAgents*[0]
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=",
                  mul_list_scalar(mean, 1 / nbTest))
            itest += 1
            for idx, ag in enumerate(agent.agents):
                logger.direct_write(
                    f"rewardTest for agent {idx}", mean[idx], i)
            logger.direct_write(f"rewardTest total", sum(mean), i)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(agent.outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        # Initialize with initial position of the agent
        new_ob = ob
        agent.step = 0

        while True:
            if verbose:
                env.render()
            ob = new_ob
            action = agent.act(ob)  # select action
            a_copy = [a_ag.detach().numpy() for a_ag in action]
            new_ob, reward, done, _ = env.step(a_copy)  # execute agent
            # list of observation for each agent
            new_ob = padObs(new_ob, mshape)
            reward = list_clip(reward, -10, 10)

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = agent.nAgents*[True]
                for ag in agent.agents:
                    ag.noise.reset()
                print("forced done!")
            agent.store(ob, action, new_ob, reward, done, j)
            rsum = add_list(rsum, reward)

            if agent.timeToLearn(done):
                _, vloss, ploss = agent.learn()

            if done[0]:
                if i % agent.freqVerbose == 0:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) +
                          " actions, value loss=" + str(vloss) + "policy loss=" + str(ploss))
                if not agent.test:
                    for idx, ag in enumerate(agent.agents):
                        logger.direct_write(
                            f"reward for agent {idx}", rsum[idx], i)
                        logger.direct_write(
                            f"value loss for agent {idx}", vloss[idx], i)
                        logger.direct_write(
                            f"policy loss for agent {idx}", ploss[idx], i)
                        ag.nbEvents = 0
                        ag.noise.reset()

                    logger.direct_write(f"reward total", sum(rsum), i)
                    logger.direct_write(f"value loss total", sum(vloss), i)
                    logger.direct_write(f"policy loss total", sum(ploss), i)
                mean = add_list(mean, rsum)
                rsum = agent.nAgents*[0]

                break

    env.close()
