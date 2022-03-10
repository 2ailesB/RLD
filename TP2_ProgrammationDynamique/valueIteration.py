"""
    This file contains the code of the Value Iteration algorithm. 
"""

import numpy as np
import gym
from gym import wrappers, logger
import gridworld
import matplotlib
import time

matplotlib.use("TkAgg")


class ValueIterationAgent(object):

    """
        Class that provides the agent that uses the Policy Itération algorithm to estimate policy.
    """

    def __init__(self, action_space):
        """
            Initiate the agent
            Input :
                action_space : Environnement action space
        """
        self.action_space = action_space

    def valueIteration(self, env, epsilon, iterMax, gamma):
        """
        Value Iteration algorithm
        Input
            self (ValueIterationAgent) : the agent
            env : the environnement (gym)
            epsilon (float) : the error for convergence of the estimation of V
            iterMax (int) : max number of iteration to compute the policy
            gamma (float) : the discount parameter for V estimation (horizon to consider)
        Output 
            pi (array) : the policy
            proba (array) : the score for each state of the policy
            errors (array) : the errors to visualize the convergence of V
        """

        err = 100
        errors = []
        states, mdp = env.getMDP()
        V = np.zeros(len(states))

        # mdp[source, action] = [(p1, s1, r1, d1), (p2, s2, r2, d2), (p3, s3, r3, d3)]
        # Loop to create V
        for i in range(iterMax):
            V_prec = V.copy()
            for s in mdp.keys():
                temp = np.zeros(len(mdp[s]))
                for a in mdp[s].keys():
                    for j in np.arange(len(mdp[s][a])):
                        sp = mdp[s][a][j][1]
                        temp[a] += mdp[s][a][j][0] * \
                            (mdp[s][a][j][2] + gamma*V_prec[sp])
                V[s] = np.max(temp)
            err = np.linalg.norm(V - V_prec)
            errors.append(err)
            if err < epsilon:
                # print(f'convergence at iteration {i}')
                break

        pi = np.zeros(len(env.states))
        proba = np.zeros(len(env.states))
        for s in mdp.keys():
            temp = np.zeros(len(mdp[s]))
            for a in mdp[s].keys():
                for j in np.arange(len(mdp[s][a])):
                    sp = mdp[s][a][j][1]
                    temp[a] += mdp[s][a][j][0]*(mdp[s][a][j][2] + gamma*V[sp])
            proba[s] = np.max(temp)
            pi[s] = np.argmax(temp)

        return pi, proba, errors


if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt",
                {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.render(mode="human")

    # Initialisation de l'agent et calcul de la politique
    agent = ValueIterationAgent(env.action_space)
    tic = time.time()
    policy, score, _ = agent.valueIteration(env, 0.01, 1000, 0.99)
    tac = time.time()
    print('politique', policy, score)
    print(f'computation time : {tac-tic}')
    
    episode_count = 100
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        agent_pos = env.getStateFromObs(obs)
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        done = False
        while not done:
            action = policy[agent_pos]
            obs, reward, done, _ = env.step(action)
            agent_pos = env.getStateFromObs(obs)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" +
                      str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
