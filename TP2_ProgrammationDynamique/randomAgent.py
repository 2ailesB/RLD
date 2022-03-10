"""
    This file contains the code to use a random agent. 
"""


import gym
from gym import wrappers, logger
import gridworld
import matplotlib

matplotlib.use("TkAgg")


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan1.txt",
                {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    print(env.step(1))
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  # visualisation sur la console
    states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
    print("Nombre d'etats : ", len(states))
    print(states)
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    print(transitions)
    
    # Execution avec un Agent
    agent = RandomAgent(env.action_space)
    
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
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
