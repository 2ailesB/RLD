"""
    This contains the code to do some tests on algorithms
"""

import numpy as np
import gym
from gym import wrappers, logger
import gridworld
import matplotlib.pyplot as plt
import time

from policyIteration import PolicyIterationAgent
from valueIteration import ValueIterationAgent

def compute_time(env, maps, nb_test, path = 'figures/'):
    """
        Compute and plot the computation time of the algorithms
        Input : 
            env : the environnement
            maps (array) : the number of the maps on which we estimate times
            nb_test (int) : number of estimation to do to compute the time per map
            path (str) : where to save plots
        Ouput :
            times (array) : the computation time for the 2 algorithms, on each map in maps
    
    """

    n_maps = len(maps)
    times = np.zeros((2, n_maps))

    for cpt, map in enumerate(maps) : 
        env.setPlan(f"gridworldPlans/plan{map}.txt",
                    {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
        print(f'maps {map}')
        env.render(mode="human")

        agentValue = ValueIterationAgent(env.action_space)
        agentPolicy = PolicyIterationAgent(env.action_space)
        
        time1mapValue = 0
        time1mapPolicy = 0
        # Initialisation de l'agent et calcul de la politique
        for i in range(nb_test) : 
            print(f'test {i}')
            print('value iteration')
            tic = time.time()
            policy, score, _ = agentValue.valueIteration(env, 0.01, 1000, 0.99)
            tac = time.time()
            time1mapValue += (tac-tic)/nb_test

            print('policy iteration')
            tic = time.time()
            policy, score, _ = agentPolicy.policyIteration(env, 0.01, 1000, 1000, 0.99)
            tac = time.time()
            time1mapPolicy += (tac-tic)/nb_test

        times[0, cpt] = time1mapValue
        times[1, cpt] = time1mapPolicy

    plt.figure()
    plt.plot(maps, times[0, :])
    plt.plot(maps, times[1, :])
    plt.yscale('log')
    plt.legend(['Value Iteration', 'Policy Iteration'])
    plt.xlabel('Map number')
    plt.ylabel('Computation Time (s)')
    plt.title(f'Comparison of Computation times for Value Iteration and policy Iteration \n Mean times computed on {nb_test} runs for each map')
    plt.savefig(f'{path}/times{nb_test}tests.jpg')

    return times

def print_policy(policy, states):
    """
        Return the policy following the geometry of the map
        Input :
            policy (array 1xn) : the policy compute from the algorithm
            states (array) : the states from the environnement
        Output : 
            policy_print (np array) : the policy with same shape than the map
    """

    policy_print = np.array(eval(states[0]), dtype=np.float)
    n, m = policy_print.shape

    for cpt, str_state in enumerate(states):
        state = np.array(eval(str_state))
        agent_pos = np.where(state == 2)
        policy_print[agent_pos] = policy[cpt]

    return policy_print[1:n-1, 1:m-1]

def compare_policies0(env, map):
    """
        Return the policy following the geometry of the map
        Input :
            env : the environnement
            map (int) : the map on which compute the policies
        Output : 
            policy_print_policy (np array) : the policy for policy iteration with same shape than the map
            policy_print_value (np array) : the policy for value iteration with same shape than the map
    """
    env.setPlan(f"gridworldPlans/plan{map}.txt",
                    {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.render(mode="human")
    states, mdp = env.getMDP()

    agentValue = ValueIterationAgent(env.action_space)
    agentPolicy = PolicyIterationAgent(env.action_space)
    policyValue, score, _ = agentValue.valueIteration(env, 0.01, 1000, 0.99)
    policyPolicy, score, _ = agentPolicy.policyIteration(env, 0.01, 1000, 1000, 0.99)

    policy_print_value = np.array(eval(states[0]))
    policy_print_policy = np.array(eval(states[0]))
    n, m = policy_print_policy.shape

    for cpt, str_state in enumerate(states):
        state = np.array(eval(str_state))
        agent_pos = np.where(state == 2)
        policy_print_value[agent_pos] = policyValue[cpt]
        policy_print_policy[agent_pos] = policyPolicy[cpt]

    return policy_print_policy[1:n-1, 1:m-1], policy_print_value[1:n-1, 1:m-1]
    
def compute_infos_value(env, map, envRewards, valueParams):
    """
        Compute informations from value iteration algorithm
        Input :
            env : the environnement
            map : the map on which we want to compute quantities
            envRewards (1x5 array): parameters of the environnement (rewards for each type of case)
            valueParams (1x3 array): parameters of the value iteration algorithm 
        Output : 
            policy_print (np array) : the policy for value iteration with same shape than the map
            scores_print (np arrat) : the scores for value iteration with same shape than the map
            errors (array) : the errors for value convergence
            len(errors) (int) : the iteration on which the algorithm has converged
    """

    env.setPlan(f"gridworldPlans/plan{map}.txt",
                    {0: envRewards[0], 3: envRewards[1], 4: envRewards[2], 5: envRewards[3], 6: envRewards[4]})
    env.render(mode="human")
    states, mdp = env.getMDP()

    agentValue = ValueIterationAgent(env.action_space)
    policyValue, score, errors = agentValue.valueIteration(env, valueParams[0], valueParams[1], valueParams[2])

    policy_print = print_policy(policyValue, states)
    scores_print = print_policy(score, states)

    return policy_print, scores_print, errors, len(errors)

def compute_infos_policy(env, map, envRewards, valueParams):
    """
        Compute informations from policy iteration algorithm
        Input :
            env : the environnement
            map : the map on which we want to compute quantities
            envRewards (1x5 array) : parameters of the environnement (rewards for each type of case)
            valueParams (1x3 array) : parameters of the policy iteration algorithm 
        Output : 
            policy_print (np array) : the policy for value iteration with same shape than the map
            scores_print (np arrat) : the scores for value iteration with same shape than the map
            iterconv (int) : the iteration on which the algorithm has converged
    """

    env.setPlan(f"gridworldPlans/plan{map}.txt",
                    {0: envRewards[0], 3: envRewards[1], 4: envRewards[2], 5: envRewards[3], 6: envRewards[4]})
    env.render(mode="human")
    states, mdp = env.getMDP()

    agentValue = PolicyIterationAgent(env.action_space)
    policyValue, score, iterconv = agentValue.policyIteration(env, valueParams[0], valueParams[1], valueParams[2], valueParams[3])

    policy_print = print_policy(policyValue, states)
    scores_print = print_policy(score, states)

    return policy_print, scores_print, iterconv


if __name__ == '__main__':

    env = gym.make("gridworld-v0")

    maps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    nb_test = 1

    compute_time(env, maps, nb_test, 'TP2_ProgrammationDynamique/figures')

    policyValue, policyPolicy = compare_policies0(env, 0)

    print('Value iteration \n', policyValue)
    print('Policy iteration \n', policyPolicy)
    print(policyValue==policyPolicy)

    # VALUE ITERATION 
    #algorithm parameters
    gammas = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1]
    epsilons = np.logspace(-4, 2, 6)
    iterMaxs = [50, 100, 500, 1000]

    print('Gammas')
    for gamma in gammas : 
        policy_print, scores_print, errors, iterconv = compute_infos_value(env, 0, [-0.001, 1, 1, -1, -1], [0.01, 1000, gamma])
        print(f'gamma : {gamma}')
        print(policy_print)
    print('epsilons')
    for epsilon in epsilons : 
        policy_print, scores_print, errors, iterconv = compute_infos_value(env, 0, [-0.001, 1, 1, -1, -1], [epsilon, 1000, 0.99])
    print('iterMax')
    for iterMax in iterMaxs : 
        policy_print, scores_print, errors, iterconv = compute_infos_value(env, 0, [-0.001, 1, 1, -1, -1], [0.01, iterMax, 0.99])

    #env parameters
    print('env modification')
    env_params = [[-0.001, 1, 1, -1, -1], [-0.1, 1, 1, -1, -1], [-0.1, 2, 1, -1, -1], [0.001, 1, 1, -1, -1], [0.001, -1, -1, -1, -1]]
    for env_param in env_params : 
        policy_print, scores_print, errors, iterconv = compute_infos_value(env, 0, env_param, [0.01, 1000, 0.99])
        print('env parameters : \n', env_param)
        print(policy_print)

    # POLICY ITERATION
    #algorithm parameters
    gammas = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1]
    epsilons = np.logspace(-4, 2, 6)
    iterMaxs = [50, 100, 500, 1000]
    iterMaxValues = [50, 100, 500, 1000]

    print('Gammas')
    for gamma in gammas : 
        policy_print, scores_print, iterconv = compute_infos_policy(env, 0, [-0.001, 1, 1, -1, -1], [0.01, 1000, 1000, gamma])
        print(f'gamma : {gamma}')
        print(policy_print)
    print('epsilons')
    for epsilon in epsilons : 
        policy_print, scores_print, iterconv = compute_infos_policy(env, 0, [-0.001, 1, 1, -1, -1], [epsilon, 1000, 1000, 0.99])
    print('iterMax')
    for iterMax in iterMaxs : 
        policy_print, scores_print, iterconv = compute_infos_policy(env, 0, [-0.001, 1, 1, -1, -1], [0.01, iterMax, 1000, 0.99])
    print('iterMaxValue')
    for iterMaxValue in iterMaxValues : 
        policy_print, scores_print, iterconv = compute_infos_policy(env, 0, [-0.001, 1, 1, -1, -1], [0.01, 1000, iterMaxValue, 0.99])

    #env parameters
    print('env modification')
    env_params = [[-0.001, 1, 1, -1, -1], [-0.1, 1, 1, -1, -1], [-0.1, 2, 1, -1, -1], [0.001, 1, 1, -1, -1], [0.001, -1, -1, -1, -1]]
    for env_param in env_params : 
        policy_print, scores_print, iterconv = compute_infos_policy(env, 0, env_param, [0.01, 1000, 1000, 0.99])
        print('env parameters : \n', env_param)
        print(policy_print)


    