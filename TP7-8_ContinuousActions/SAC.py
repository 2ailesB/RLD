"""
    This file contains functions and script to execute the SAC algorithm. 
    For more informations on algorithms see https://dac.lip6.fr/master/rld-2021-2022/.
"""

from memory import *
from core import *
from utils import *
import torch
from gym.core import ObservationWrapper
import matplotlib
import copy
from torch.utils.tensorboard import SummaryWriter

from torch.distributions.transforms import SigmoidTransform
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")


class SAC(object):
    """
        class that defines the agent using SAC to play
    """

    def __init__(self, env, opt):

        #env params
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        # Now with continuous actions, the first elt is the lowest acepted value and the second the highest
        self.action_space = env.action_space
        # number of possible actions
        self.action_space_n = self.action_space.shape[0]
        self.obs_space = env.observation_space
        self.obs_space_n = env.observation_space.shape[0]
        self.featureExtractor = opt.featExtractor(env)
        self.test = False
        self.nbEvents = 0

        # Hyper parameters for learning
        self.gamma = opt.gamma  # parameter for computation of y

        # Memory
        # mémoire de batches qui contient self.mem_size transitions transitions
        self.mem_size = opt.mem_size
        self.mbs = opt.mbs  # taille des minibatch pour l'apprentissage /!\ mbs <=mem_size
        self.memory = Memory(self.mem_size)  # Memory
        self.norm_rew = opt.norm_rew # Normalize the rewards

        # Networks
        # Q doit dépendre des actions et des état ==> concaténer + modifier le réseau
        self.Qvalue1 = NN(self.featureExtractor.outSize + self.action_space_n, 1, layers=opt.QValueLayers1, finalActivation=None, activation=F.leaky_relu)  # first NN : learn
        self.Qvalue1target = copy.deepcopy(self.Qvalue1)
        self.Qvalue2 = NN(self.featureExtractor.outSize + self.action_space_n, 1, layers=opt.QValueLayers2, finalActivation=None, activation=F.leaky_relu)  # first NN : learn
        self.Qvalue2target = copy.deepcopy(self.Qvalue2)
        self.policy = NN(self.featureExtractor.outSize, 2 * self.action_space_n, layers=opt.ActorLayers, finalActivation=None, activation=F.leaky_relu)  # first NN : learn
        self.C = opt.C  # update target each C steps
        self.evts = 0  # evts
        self.freqOptim = opt.freqOptim
        self.freqVerbose = opt.freqVerbose
        # learning rate for value function = Q function not V /!\ to change for more readability
        self.lrValue = opt.ValueLr
        self.lrPolicy = opt.ActorLr # lr for policy
        self.Qvalue1Optimizer = torch.optim.Adam(
            self.Qvalue1.parameters(), lr=self.lrValue)
        self.Qvalue2Optimizer = torch.optim.Adam(
            self.Qvalue2.parameters(), lr=self.lrValue)
        self.policyOptimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lrPolicy)
        self.Qvalue1Loss = F.mse_loss
        self.Qvalue2Loss = F.mse_loss
        self.policyLoss = F.smooth_l1_loss
        self.tau = opt.tau

        # Noise for act
        self.noise = torch.distributions.Normal(0, 1)
        self.alpha = opt.alpha

    def getAction(self, state):
        """
            This function choose the next action using the policy and the noise of the agent
            Input : 
                state : the currunt state
            Output : 
                action (torch tensor) : the chosen actions
                prob (torch tensor) : the probability for each action
        """
        minval = torch.from_numpy(self.action_space.low)
        maxval = torch.from_numpy(self.action_space.high)

        # 1 action per state of the batch
        x = self.noise.rsample((state.size()[0], self.action_space_n))
        prob = self.noise.cdf(x)
        musigma_st = self.policy.forward(state)  # bug's here
        # Les n premiers sont les moyennes et les n derniers les std
        x = x * musigma_st[:, 0:self.action_space_n] + \
            musigma_st[:, self.action_space_n:2 * self.action_space_n]
        action = (1 + F.tanh(x)) * (maxval - minval) / 2 + minval

        return action, prob

    def act(self, obs):
        """
            Find action by sampling from the probability of the actor + a chosen noise
            Input :
                obs : the state of the agent
            Output : 
                action : The actions to be played
        """

        action, _ = self.getAction(torch.from_numpy(obs).float())

        return action

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
                qloss1 : the loss of the 1st qnet
                qloss2 : the loss of the 2nd q net
                policyl : the loss of the policy
        """
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return self, 0, 0, 0

        #Si la mémoire n'est pas assez remplie, on n'apprend pas
        if self.memory.nentities <= self.mbs:
            print('memory not enough filled ', self.memory.nentities,
                  'available of out ', self.mbs, 'needed for 1 batch')
            return self, 0, 0, 0

        # sinon on calcule y et on fait une descente de gradient
        # On tire aléatoirement un mini batch de transitions
        chosenIdx, z, chosentr = self.memory.sample(self.mbs)
        # 1 transition  = (départ (list), actions(int), reward(float), arrivée(list), done(bool))
        # conversion in torch tensor
        starts = torch.Tensor([tr[0] for tr in chosentr]).squeeze()
        # actions = torch.Tensor([tr[1] for tr in chosentr]).unsqueeze(-1)
        actions = torch.Tensor([tr[1].detach().numpy() for tr in chosentr]).view(self.mbs, self.action_space_n)
        rewards = torch.Tensor([tr[2] for tr in chosentr]).unsqueeze(-1)
        dests = torch.Tensor([tr[3] for tr in chosentr]).squeeze()
        dones = torch.Tensor([tr[4] for tr in chosentr]).unsqueeze(-1)

        #STEP 12
        #compute yq
        with torch.no_grad():
            adests, probdest = self.getAction(dests)
            dests_adests = torch.cat((dests, adests), 1)
            Q1targ = self.Qvalue1target(dests_adests).detach()
            Q2targ = self.Qvalue2target(dests_adests).detach()
            Qtarg = torch.min(Q1targ, Q2targ)
            yq = rewards + (1 - dones) * \
                (Qtarg - self.alpha * torch.log(probdest)).detach()

            

            astarts, probstarts = self.getAction(starts)
            starts_astarts = torch.cat((starts, astarts), 1)
            Q1_pred = self.Qvalue1(starts_astarts)
            Q2_pred = self.Qvalue2(starts_astarts)
            yv = torch.min(Q1_pred, Q2_pred) - \
                self.alpha * torch.log(probstarts)

        # STEP 13
        #update of the Qnets
        self.Qvalue1Optimizer.zero_grad()
        starts_actions = torch.cat((starts, actions), 1)
        Q1_obs = self.Qvalue1(starts_actions)
        qloss1 = self.Qvalue1Loss(yq, Q1_obs)  # /self.mbs
        qloss1.backward()
        self.Qvalue1Optimizer.step()
        # ou faire une loss qui soit la somme des deux ?

        self.Qvalue2Optimizer.zero_grad()
        starts_actions = torch.cat((starts, actions), 1)
        Q2_obs = self.Qvalue2(starts_actions)
        qloss2 = self.Qvalue2Loss(yq, Q2_obs)  # /self.mbs
        qloss2.backward()
        self.Qvalue2Optimizer.step()


        # STEP 15
        self.policyOptimizer.zero_grad()
        astarts, probstarts = self.getAction(starts)
        starts_astarts = torch.cat((starts, astarts), 1)
        Q1_pred = self.Qvalue1(starts_astarts)  # or use min ?
        policyl = - (Q1_pred - self.alpha * torch.log(probstarts)).mean() 
        policyl.backward()
        self.policyOptimizer.step()

        # each C steps, update Qhat net
        if self.evts % self.C == 0:
            self.soft_update(self.Qvalue1, self.Qvalue1target, self.tau)
            self.soft_update(self.Qvalue2, self.Qvalue2target, self.tau)
        
        self.evts += 1
        #a
        return self, qloss1, qloss2, policyl

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward/self.norm_rew, new_ob, done)
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
        './TP7-8_ContinuousActions/configs/config_pendulum-v0_SAC.yaml', "SACAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = SAC(env, config)

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
                print("forced done!")
            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward

            if agent.timeToLearn(done):
                _, q1loss, q2loss, ploss = agent.learn()
                
            if done:
                if i % agent.freqVerbose == 0:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions, Q1net loss=" +
                          str(q1loss) + " Q2net loss=" + str(q2loss) + " policy loss=" + str(ploss))
                logger.direct_write("reward", rsum, i)
                logger.direct_write("Qnet1 loss", q1loss, i)
                logger.direct_write("Qnet2 loss", q2loss, i)
                logger.direct_write("policy loss", ploss, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
