# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:21:16 2024

@author: lily
"""
import numpy as np
import copy

def solve_agent_distribution(N_type, urgent_prob, agent_distribution_equal):
    if N_type == 1:
        agent_distribution = [1,0,0,0,0]
    else:
        if agent_distribution_equal == 1:
            agent_distribution = [0.2]*5
        else:
            p1 = urgent_prob[0]
            p2 = urgent_prob[1]
            p3 = urgent_prob[2]
            p4 = urgent_prob[3]
            p5 = urgent_prob[4]
            A = np.array([[1-p1, p2*p3*p4*p5, p3*p4*p5*(1-p1), p4*p5*(1-p1), p5*(1-p1)],
                          [p1*(1-p2), 1-p2, p3*p4*p5*p1, p4*p5*p1*(1-p2), p5*p1*(1-p2)],
                          [p1*p2*(1-p3), p2*(1-p3), 1-p3, p4*p5*p1*p2, p5*p1*p2*(1-p3)],
                          [p1*p2*p3*(1-p4), p2*p3*(1-p4), p3*(1-p4), 1-p4, p5*p1*p2*p3],
                          [p1*p2*p3*p4, p2*p3*p4*(1-p5), p3*p4*(1-p5), p4*(1-p5), 1-p5]])
            B = np.array([0.2,0.2,0.2,0.2,0.2])
            X = np.linalg.solve(A, B)
            for i in range(N_day): # update X, remove negative values
                if X[i] < 0:
                    X[i] = 0                
            agent_distribution = [0]*N_day
            for i in range(N_day): # get agent population based on updated X
                agent_distribution[i] = X[i] / np.sum(X)        
    return agent_distribution
 
def get_N_car_agent(N_type, agent_distribution, population_pi, self_type): #1st dim: bus_already, 2nd dim: day, 3rd dim: urgent, 4th dim: action
    N, N1b, N2b, N3b, N4b, N5b = [0]*N_type, [0]*N_type, [0]*N_type, [0]*N_type, [0]*N_type, [0]*N_type
    for a_type, pi_temp in population_pi.items():
        if self_type == a_type: # if self type, if self_type == None, all types are other types
            N[a_type] = N_agent * agent_distribution[a_type] - 1
        else: # if other types
            N[a_type] = N_agent * agent_distribution[a_type]
        if N_type == 1:
            # for each type, the number of agent taking bus on 1st day, is the one taking bus if urgent + the one taking bus if non-urgent
            N1b[a_type] = N[a_type] * (pi_temp[0][0][0][1] * (1-urgent_prob[0]) + pi_temp[0][0][1][1] * urgent_prob[0])
            # the number of agent taking bus on 2nd day, is the one taking bus if urgent + the one taking bus if non-urgent
            N2b[a_type] = (N[a_type] - N1b[a_type]) * (pi_temp[0][1][0][1] * (1-urgent_prob[1]) + pi_temp[0][1][1][1] * urgent_prob[1])
            #..
            N3b[a_type] = (N[a_type] - N1b[a_type] - N2b[a_type]) * (pi_temp[0][2][0][1] * (1-urgent_prob[2]) + pi_temp[0][2][1][1] * urgent_prob[2])
            N4b[a_type] = (N[a_type] - N1b[a_type] - N2b[a_type] - N3b[a_type]) * (pi_temp[0][3][0][1] * (1-urgent_prob[3]) + pi_temp[0][3][1][1] * urgent_prob[3])
        elif N_type == 5:
            # for each type, the number of agent taking bus on 1st day, is the one taking bus if urgent + the one taking bus if non-urgent
            # N1b stores the nunmber of agent taking bus on the first day (not Monday) of each type, e.g., the first of type 0 is Monday, of type 1 is Tuesday
            N1b[a_type] = N[a_type] * (pi_temp[0][0][0][1] * (1-urgent_prob[(0+a_type)%N_type]) + pi_temp[0][0][1][1] * urgent_prob[(0+a_type)%N_type])
            # the number of agent taking bus on 2nd day, is the one taking bus if urgent + the one taking bus if non-urgent
            N2b[a_type] = (N[a_type] - N1b[a_type]) * (pi_temp[0][1][0][1] * (1-urgent_prob[(1+a_type)%N_type]) + pi_temp[0][1][1][1] * urgent_prob[(1+a_type)%N_type])
            #..
            N3b[a_type] = (N[a_type] - N1b[a_type] - N2b[a_type]) * (pi_temp[0][2][0][1] * (1-urgent_prob[(2+a_type)%N_type]) + pi_temp[0][2][1][1] * urgent_prob[(2+a_type)%N_type])
            N4b[a_type] = (N[a_type] - N1b[a_type] - N2b[a_type] - N3b[a_type]) * (pi_temp[0][3][0][1] * (1-urgent_prob[(3+a_type)%N_type]) + pi_temp[0][3][1][1] * urgent_prob[(3+a_type)%N_type])
        # the number of agent taking bus on 5th (last) day, is the number of agents that have not taking bus
        N5b[a_type] = N[a_type] - N1b[a_type] - N2b[a_type] - N3b[a_type] -N4b[a_type]
    
    if N_type == 1:
        n1b = N1b[0]
        n2b = N2b[0]
        n3b = N3b[0]
        n4b = N4b[0]
        n5b = N5b[0]
        Ncar_dic = {}
        if self_type == None:
            Ncar_dic['full'] = [N_agent-n1b, N_agent-n2b, N_agent-n3b, N_agent-n4b, N_agent-n5b]
        else:
            Ncar_dic['full'] = [N_agent-1-n1b, N_agent-1-n2b, N_agent-1-n3b, N_agent-1-n4b, N_agent-1-n5b]
        Ncar_dic[0] = [N[0]-N1b[0], N[0]-N2b[0], N[0]-N3b[0], N[0]-N4b[0], N[0]-N5b[0]]
        Ncar_dic['self_order'] = {}
        Ncar_dic['self_order'][0] = [N[0]-N1b[0], N[0]-N2b[0], N[0]-N3b[0], N[0]-N4b[0], N[0]-N5b[0]] # for type 0: the first day is Monday, the last day is Friday
    elif N_type == 5:
        n1b = N1b[0] + N2b[4] + N3b[3] + N4b[2] + N5b[1]
        n2b = N1b[1] + N2b[0] + N3b[4] + N4b[3] + N5b[2]
        n3b = N1b[2] + N2b[1] + N3b[0] + N4b[4] + N5b[3]
        n4b = N1b[3] + N2b[2] + N3b[1] + N4b[0] + N5b[4]
        n5b = N1b[4] + N2b[3] + N3b[2] + N4b[1] + N5b[0]
        Ncar_dic = {}
        if self_type == None:
            Ncar_dic['full'] = [N_agent-n1b, N_agent-n2b, N_agent-n3b, N_agent-n4b, N_agent-n5b]
        else:
            Ncar_dic['full'] = [N_agent-1-n1b, N_agent-1-n2b, N_agent-1-n3b, N_agent-1-n4b, N_agent-1-n5b]
        # Ncar_dic[type] stores the number of "type" agent taking car from Monday to Friday, in order
        Ncar_dic[0] = [N[0]-N1b[0], N[0]-N2b[0], N[0]-N3b[0], N[0]-N4b[0], N[0]-N5b[0]] # for type 0: the first day is Monday, the last day is Friday
        Ncar_dic[1] = [N[1]-N5b[1], N[1]-N1b[1], N[1]-N2b[1], N[1]-N3b[1], N[1]-N4b[1]] # for type 1: the first day is Tuesday, the last day is Monday
        Ncar_dic[2] = [N[2]-N4b[2], N[2]-N5b[2], N[2]-N1b[2], N[2]-N2b[2], N[2]-N3b[2]] # for type 1: the first day is Wednesday, the last day is Tuesday
        Ncar_dic[3] = [N[3]-N3b[3], N[3]-N4b[3], N[3]-N5b[3], N[3]-N1b[3], N[3]-N2b[3]] # for type 1: the first day is Thursday, the last day is Wednesday
        Ncar_dic[4] = [N[4]-N2b[4], N[4]-N3b[4], N[4]-N4b[4], N[4]-N5b[4], N[4]-N1b[4]] # for type 1: the first day is Friday, the last day is Thursday
        # Ncar_dic['self_order'][type] stores the number of "type" agent taking car from its first day (maybe not Monday) to last day
        Ncar_dic['self_order'] = {}
        Ncar_dic['self_order'][0] = [N[0]-N1b[0], N[0]-N2b[0], N[0]-N3b[0], N[0]-N4b[0], N[0]-N5b[0]] # for type 0: the first day is Monday, the last day is Friday
        Ncar_dic['self_order'][1] = [N[1]-N1b[1], N[1]-N2b[1], N[1]-N3b[1], N[1]-N4b[1], N[1]-N5b[1]] # for type 1: the first day is Tuesday, the last day is Monday
        Ncar_dic['self_order'][2] = [N[2]-N1b[2], N[2]-N2b[2], N[2]-N3b[2], N[2]-N4b[2], N[2]-N5b[2]] # for type 1: the first day is Wednesday, the last day is Tuesday
        Ncar_dic['self_order'][3] = [N[3]-N1b[3], N[3]-N2b[3], N[3]-N3b[3], N[3]-N4b[3], N[3]-N5b[3]] # for type 1: the first day is Thursday, the last day is Wednesday
        Ncar_dic['self_order'][4] = [N[4]-N1b[4], N[4]-N2b[4], N[4]-N3b[4], N[4]-N4b[4], N[4]-N5b[4]] # for type 1: the first day is Friday, the last day is Thursday
    return Ncar_dic
    
def initialize_state_action_distribution(N_type, agent_distribution, population_pi):
    #L: 1st dim: bus_already, 2nd dim: day, 3rd dim: urgent, 4th dim: action
    state_distribution = {}
    state_action_distribution = {}
    for a_type in range(N_type):
        Ncar_dic = get_N_car_agent(N_type, agent_distribution, population_pi,self_type=None)['self_order'][a_type] # if self_type=None, include the representative agent in population
        N = N_agent * agent_distribution[a_type]
        n1b = N - Ncar_dic[0] # the number of "type" agent taking bus on its first day
        n2b = N - Ncar_dic[1]
        n3b = N - Ncar_dic[2]
        n4b = N - Ncar_dic[3]
        # 1/5: every day is of equal prob, urgent_prob for urgent, (1-urgent_prob) for non-urgent
        # for bus_already_prob: the cumulative number of agents already taken bus / N
        if N_type == 5:
            if N > 0:
                state_distribution[a_type] = np.array([[[1/5 * (1 - urgent_prob[(0+a_type)%N_type]) * 1, 1/5 * urgent_prob[(0+a_type)%N_type] * 1],
                                                [1/5 * (1 - urgent_prob[(1+a_type)%N_type]) * (1 - n1b / N), 1/5 * urgent_prob[(1+a_type)%N_type] * (1 - n1b / N)],
                                                [1/5 * (1 - urgent_prob[(2+a_type)%N_type]) * (1 - (n1b + n2b) / N), 1/5 * urgent_prob[(2+a_type)%N_type] * (1 - (n1b + n2b) / N)],
                                                [1/5 * (1 - urgent_prob[(3+a_type)%N_type]) * (1 - (n1b + n2b + n3b) / N), 1/5 * urgent_prob[(3+a_type)%N_type] * (1 - (n1b + n2b + n3b) / N)],
                                                [1/5 * (1 - urgent_prob[(4+a_type)%N_type]) * (1 - (n1b + n2b + n3b +n4b) / N), 1/5 * urgent_prob[(4+a_type)%N_type] * (1 - (n1b + n2b + n3b +n4b) / N)]],
                                    
                                               [[1/5 * (1 - urgent_prob[(0+a_type)%N_type]) * 0, 1/5 * urgent_prob[(0+a_type)%N_type] * 0],
                                                [1/5 * (1 - urgent_prob[(1+a_type)%N_type]) * n1b / N, 1/5 * urgent_prob[(1+a_type)%N_type] * n1b / N],
                                                [1/5 * (1 - urgent_prob[(2+a_type)%N_type]) * (n1b + n2b) / N, 1/5 * urgent_prob[(2+a_type)%N_type] * (n1b + n2b) / N],
                                                [1/5 * (1 - urgent_prob[(3+a_type)%N_type]) * (n1b + n2b + n3b) / N, 1/5 * urgent_prob[(3+a_type)%N_type] * (n1b + n2b + n3b) / N],
                                                [1/5 * (1 - urgent_prob[(4+a_type)%N_type]) * (n1b + n2b + n3b +n4b) / N, 1/5 * urgent_prob[(4+a_type)%N_type] * (n1b + n2b + n3b +n4b) / N]]])
            elif N == 0:
                state_distribution[a_type] = np.array([[[0, 0],
                                                        [0, 0],
                                                        [0, 0],
                                                        [0, 0],
                                                        [0, 0]],
                                            
                                                       [[0, 0],
                                                        [0, 0],
                                                        [0, 0],
                                                        [0, 0],
                                                        [0, 0]]])
        elif N_type == 1:
            state_distribution[a_type] = np.array([[[1/5 * (1 - urgent_prob[0]) * 1, 1/5 * urgent_prob[0] * 1],
                                            [1/5 * (1 - urgent_prob[1]) * (1 - n1b / N), 1/5 * urgent_prob[1] * (1 - n1b / N)],
                                            [1/5 * (1 - urgent_prob[2]) * (1 - (n1b + n2b) / N), 1/5 * urgent_prob[2] * (1 - (n1b + n2b) / N)],
                                            [1/5 * (1 - urgent_prob[3]) * (1 - (n1b + n2b + n3b) / N), 1/5 * urgent_prob[3] * (1 - (n1b + n2b + n3b) / N)],
                                            [1/5 * (1 - urgent_prob[4]) * (1 - (n1b + n2b + n3b +n4b) / N), 1/5 * urgent_prob[4] * (1 - (n1b + n2b + n3b +n4b) / N)]],
                                
                                           [[1/5 * (1 - urgent_prob[0]) * 0, 1/5 * urgent_prob[0] * 0],
                                            [1/5 * (1 - urgent_prob[1]) * n1b / N, 1/5 * urgent_prob[1] * n1b / N],
                                            [1/5 * (1 - urgent_prob[2]) * (n1b + n2b) / N, 1/5 * urgent_prob[2] * (n1b + n2b) / N],
                                            [1/5 * (1 - urgent_prob[3]) * (n1b + n2b + n3b) / N, 1/5 * urgent_prob[3] * (n1b + n2b + n3b) / N],
                                            [1/5 * (1 - urgent_prob[4]) * (n1b + n2b + n3b +n4b) / N, 1/5 * urgent_prob[4] * (n1b + n2b + n3b +n4b) / N]]])
        state_action_distribution_temp = np.zeros([N_bus_already, N_day, N_urgent, N_action])
        for i in range(N_bus_already):
            for j in range(N_day):
                for k in range(N_urgent):
                    for m in range(N_action):
                        if truncation == 1:
                            state_action_distribution_temp[i][j][k][m] = int(state_distribution[a_type][i][j][k] * population_pi[a_type][i][j][k][m]*10000)/10000  # truncate the distribution to 4 digits
                        else:
                            state_action_distribution_temp[i][j][k][m] = state_distribution[a_type][i][j][k] * population_pi[a_type][i][j][k][m]
        state_action_distribution[a_type] = copy.deepcopy(state_action_distribution_temp)
    return state_distribution, state_action_distribution

def update_state_action_distribution(N_type, state_distribution,population_pi):
    new_state_distribution = {}
    new_state_action_distribution = {}
    for a_type in range(N_type):        
        new_state_distribution_temp = np.zeros([N_bus_already, N_day, N_urgent])
        temp_state_distribution = np.zeros([N_bus_already, N_day, N_urgent])
        new_state_action_distribution_temp = np.zeros([N_bus_already, N_day, N_urgent, N_action])
        if N_type == 5:
            for j in [0, 1, 2, 3]: # update the 2nd, 3rd, 4th and 5th day first
                temp_state_distribution[0][j+1][0] = state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][0] * (1-urgent_prob[(j+1+a_type)%N_type]) + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][0] * (1-urgent_prob[(j+1+a_type)%N_type])
                temp_state_distribution[0][j+1][1] = state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][0] * urgent_prob[(j+1+a_type)%N_type] + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][0] * urgent_prob[(j+1+a_type)%N_type]
               
                temp_state_distribution[1][j+1][0] = state_distribution[a_type][1][j][0] * (1-urgent_prob[(j+1+a_type)%N_type]) + \
                                                     state_distribution[a_type][1][j][1] * (1-urgent_prob[(j+1+a_type)%N_type]) + \
                                                     state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][1] * (1-urgent_prob[(j+1+a_type)%N_type]) + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][1] * (1-urgent_prob[(j+1+a_type)%N_type])
                temp_state_distribution[1][j+1][1] = state_distribution[a_type][1][j][0] * urgent_prob[(j+1+a_type)%N_type] + \
                                                     state_distribution[a_type][1][j][1] * urgent_prob[(j+1+a_type)%N_type] + \
                                                     state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][1] * urgent_prob[(j+1+a_type)%N_type] + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][1] * urgent_prob[(j+1+a_type)%N_type]
             
            temp_state_distribution[0][0][0] = state_distribution[a_type][0][4][0] * (1 - urgent_prob[(0+a_type)%N_type]) + \
                                               state_distribution[a_type][0][4][1] * (1 - urgent_prob[(0+a_type)%N_type]) + \
                                               state_distribution[a_type][1][4][0] * (1 - urgent_prob[(0+a_type)%N_type]) + \
                                               state_distribution[a_type][1][4][1] * (1 - urgent_prob[(0+a_type)%N_type])
                                              
            temp_state_distribution[0][0][1] = state_distribution[a_type][0][4][0] * urgent_prob[(0+a_type)%N_type] + \
                                               state_distribution[a_type][0][4][1] * urgent_prob[(0+a_type)%N_type] + \
                                               state_distribution[a_type][1][4][0] * urgent_prob[(0+a_type)%N_type] + \
                                               state_distribution[a_type][1][4][1] * urgent_prob[(0+a_type)%N_type]
        elif N_type == 1:
            for j in [0, 1, 2, 3]: # update the 2nd, 3rd, 4th and 5th day first
                temp_state_distribution[0][j+1][0] = state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][0] * (1-urgent_prob[j+1]) + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][0] * (1-urgent_prob[j+1])
                temp_state_distribution[0][j+1][1] = state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][0] * urgent_prob[j+1] + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][0] * urgent_prob[j+1]
               
                temp_state_distribution[1][j+1][0] = state_distribution[a_type][1][j][0] * (1-urgent_prob[j+1]) + \
                                                     state_distribution[a_type][1][j][1] * (1-urgent_prob[j+1]) + \
                                                     state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][1] * (1-urgent_prob[j+1]) + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][1] * (1-urgent_prob[j+1])
                temp_state_distribution[1][j+1][1] = state_distribution[a_type][1][j][0] * urgent_prob[j+1] + \
                                                     state_distribution[a_type][1][j][1] * urgent_prob[j+1] + \
                                                     state_distribution[a_type][0][j][0] * population_pi[a_type][0][j][0][1] * urgent_prob[j+1] + \
                                                     state_distribution[a_type][0][j][1] * population_pi[a_type][0][j][1][1] * urgent_prob[j+1]
             
            temp_state_distribution[0][0][0] = state_distribution[a_type][0][4][0] * (1 - urgent_prob[0]) + \
                                               state_distribution[a_type][0][4][1] * (1 - urgent_prob[0]) + \
                                               state_distribution[a_type][1][4][0] * (1 - urgent_prob[0]) + \
                                               state_distribution[a_type][1][4][1] * (1 - urgent_prob[0])
                                              
            temp_state_distribution[0][0][1] = state_distribution[a_type][0][4][0] * urgent_prob[0] + \
                                               state_distribution[a_type][0][4][1] * urgent_prob[0] + \
                                               state_distribution[a_type][1][4][0] * urgent_prob[0] + \
                                               state_distribution[a_type][1][4][1] * urgent_prob[0]
        
        for i in range(N_bus_already):
            for j in range(N_day):
                for k in range(N_urgent):
                    if np.sum(temp_state_distribution) > 0:
                        new_state_distribution_temp[i][j][k] = temp_state_distribution[i][j][k] / np.sum(temp_state_distribution)
                    else:
                        new_state_distribution_temp[i][j][k] = 0
        
        for i in range(N_bus_already):
            for j in range(N_day):
                for k in range(N_urgent):
                    #new_state_distribution[i][j][k] = int(new_state_distribution[i][j][k] * 10000)/10000
                    for m in range(N_action):
                        if truncation == 1:
                            new_state_action_distribution_temp[i][j][k][m] = int(new_state_distribution_temp[i][j][k] * population_pi[a_type][i][j][k][m]*10000)/10000 # truncate the distribution to 4 digits
                        else:
                            new_state_action_distribution_temp[i][j][k][m] = new_state_distribution_temp[i][j][k] * population_pi[a_type][i][j][k][m]
        new_state_distribution[a_type] = new_state_distribution_temp
        new_state_action_distribution[a_type] = new_state_action_distribution_temp
    return new_state_distribution, new_state_action_distribution


def initialize_Q():
    Q = np.full([2,5,2,2], -100, dtype=float)
    for i in range(N_bus_already):
        for j in range(N_day):
            for k in range(N_urgent):
                if i == 1: # already taken bus
                    Q[i][j][k][1] = -200 # must take car
                elif i == 0 and j == 4: # have not taken bus and on the last day
                    Q[i][j][k][0] = -200 # must take bus
    return Q

def generate_pi(car_prob):
    if car_prob == None:
        p1 = np.random.random()
        p2 = np.random.random()
        p3 = np.random.random()
        p4 = np.random.random()
        pi_temp = np.array( #1st dim: bus_already, 2nd dim: day, 3rd dim: urgent, 4th dim: action
            # if have not taken bus, then take bus with 0.5 prob on the first 4 days, and take bus with 1 prob on the last day
                        [[[[p1, 1-p1],
                           [1, 0]],
                    
                          [[p2, 1-p2],
                           [1, 0]],
                    
                          [[p3, 1-p3],
                           [1, 0]],
                    
                          [[p4, 1-p4],
                           [1, 0]],
                    
                          [[0, 1],
                           [0, 1]]],
                    
                         # if already take bus, the action should be take car regardless of other state parameters
                         [[[1, 0],
                          [1, 0]],
                    
                          [[1, 0],
                           [1, 0]],
                    
                          [[1, 0],
                           [1, 0]],
                    
                          [[1, 0],
                           [1, 0]],
                    
                          [[1, 0],
                           [1, 0]]]], dtype=np.float64)
    else:
        pi_temp = np.array( #1st dim: bus_already, 2nd dim: day, 3rd dim: urgent, 4th dim: action
            # if have not taken bus, then take bus with 0.5 prob on the first 4 days, and take bus with 1 prob on the last day
                        [[[[car_prob, 1-car_prob],
                           [1, 0]],
                    
                          [[car_prob, 1-car_prob],
                           [1, 0]],
                    
                          [[car_prob, 1-car_prob],
                           [1, 0]],
                    
                          [[car_prob, 1-car_prob],
                           [1, 0]],
                    
                          [[0, 1],
                           [0, 1]]],
                    
                         # if already take bus, the action should be take car regardless of other state parameters
                         [[[1, 0],
                          [1, 0]],
                    
                          [[1, 0],
                           [1, 0]],
                    
                          [[1, 0],
                           [1, 0]],
                    
                          [[1, 0],
                           [1, 0]],
                    
                          [[1, 0],
                           [1, 0]]]], dtype=np.float64)
    return pi_temp
    
def initialize_pi(N_type, homogenerous_pi0, pc):    
    pi0 = {}
    for i in range(N_type):
        if homogenerous_pi0 == 1: # generate same pi0 for different agent types
            pi_temp = generate_pi(car_prob=pc)
        else:
            pi_temp = generate_pi(car_prob=None)
        pi0[i] = copy.deepcopy(pi_temp)
    return pi0

def Value_iteration(N_type, agent_distribution, Q_epochs,population_pi,epoch,a_type):
    new_Q = initialize_Q()
    N = N_agent * agent_distribution[a_type]
    if N > 0: # if this type of agent is not empty
        car_list = get_N_car_agent(N_type, agent_distribution, population_pi, self_type=None)['full'] # return the full car number distribution, excluding representative agents
        for Q_epoch in range(Q_epochs):
            Q_temp = copy.deepcopy(new_Q)
            q_diff = 0
            for i in range(N_bus_already):
                for j in range(N_day):
                    for k in range(N_urgent):
                        for m in range(N_action):
                            if i == 0 and j < 4 and k == 0: # if have not taken bus, not the last day, not urgent
                                if m == 0: # if take car
                                    #car_cost = cons + a * [max(car_no - m0, 0)] ** e
                                    car_no = car_list[(j+a_type)%N_type] if N_type == 5 else car_list[j]
                                    reward = cons + a * max(car_no - m0, 0) ** e
                                    next_state = [[i,j+1,0],[i,j+1,1]]
                                    next_state_prob = [1-urgent_prob[(j+1+a_type)%N_type], urgent_prob[(j+1+a_type)%N_type]] if N_type == 5 else [1-urgent_prob[j+1], urgent_prob[j+1]]    
                                else: #if take bus
                                    reward = c2
                                    next_state = [[i+1,j+1,0],[i+1,j+1,1]]
                                    next_state_prob = [1-urgent_prob[(j+1+a_type)%N_type], urgent_prob[(j+1+a_type)%N_type]] if N_type == 5 else [1-urgent_prob[j+1], urgent_prob[j+1]]  
                                target = 0
                                for n in range(len(next_state)):
                                    s = next_state[n]
                                    target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                new_Q[i][j][k][m] = reward + gamma * target
                            elif i == 0 and j < 4 and k == 1: # if have not taken bus, not the last day, urgent
                                if m == 0: # if take car
                                    car_no = car_list[(j+a_type)%N_type] if N_type == 5 else car_list[j]
                                    reward = cons + a * max(car_no - m0, 0) ** e
                                    next_state = [[i,j+1,0],[i,j+1,1]]
                                    next_state_prob = [1-urgent_prob[(j+1+a_type)%N_type], urgent_prob[(j+1+a_type)%N_type]] if N_type == 5 else [1-urgent_prob[j+1], urgent_prob[j+1]]  
                                    target = 0
                                    for n in range(len(next_state)):
                                        s = next_state[n]
                                        target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                    new_Q[i][j][k][m] = reward + gamma * target
                            elif i == 1: # already taken bus
                                if m == 0:
                                    car_no = car_list[(j+a_type)%N_type] if N_type == 5 else car_list[j]
                                    reward = cons + a * max(car_no - m0, 0) ** e
                                    target = 0
                                    if j < 4: # not last day
                                        next_state = [[i,j+1,0],[i,j+1,1]]
                                        next_state_prob = [1-urgent_prob[(j+1+a_type)%N_type], urgent_prob[(j+1+a_type)%N_type]] if N_type == 5 else [1-urgent_prob[j+1], urgent_prob[j+1]]  
                                        for n in range(len(next_state)):
                                            s = next_state[n]
                                            target += Q_temp[s[0]][s[1]][s[2]].max() * next_state_prob[n]
                                    new_Q[i][j][k][m] = reward + gamma * target
                            elif i == 0 and j == 4:# if have not taken bus, the last day
                                if k == 0 and m == 1:
                                    reward = c2
                                    target = 0
                                    new_Q[i][j][k][m] = reward + gamma * target
                                elif k == 1 and m == 1:
                                    reward = c3
                                    target = 0
                                    new_Q[i][j][k][m] = reward + gamma * target
                            q_diff += (new_Q[i][j][k][m] - Q_temp[i][j][k][m])**2
            if q_diff == 0:
                #print(Q_epoch,'Converged!')
                break    
    reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp = get_reward(N_type, agent_distribution, population_pi, 0, new_Q, a_type)
    
    return reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp, new_Q

def get_reward(N_type, agent_distribution, population_pi, theory, new_Q, a_type):
    if theory == 'PRP':
        car_list_full = [N_agent*0.8]*5
        car_list_self = [N_agent*0.8]*5
    else:
        car_list = get_N_car_agent(N_type, agent_distribution, population_pi,self_type=None)
        car_list_full = car_list['full'] # get the full car number ditribution, including representative agents
        car_list_self = car_list['self_order'][a_type]
    N = N_agent * agent_distribution[a_type]
    reward_sum = 0
    drive_cost_sum = 0
    transit_cost_sum = 0
    if N > 0:
        for i in range(5): # 5 days
            if theory != 'PRP' and i < 4:
                if N_type == 5:
                    car_no = car_list_full[(i+a_type)%N_type]
                    car_cost = cons + a * max(car_no - m0, 0) ** e
                    reward_sum += c2 * (N - car_list_self[i]) + car_cost * car_list_self[i]
                    drive_cost_sum += car_cost * car_list_self[i]
                    transit_cost_sum += c2 * (N - car_list_self[i])
                elif N_type == 1:
                    car_no = car_list_full[i]
                    car_cost = cons + a * max(car_no - m0, 0) ** e
                    reward_sum += c2 * (N - car_list_self[i]) + car_cost * car_list_self[i]
                    drive_cost_sum += car_cost * car_list_self[i]
                    transit_cost_sum += c2 * (N - car_list_self[i])
            else: # i == 4 or theory == 'PRP'
                if N_type == 5:
                    car_no = car_list_full[(i+a_type)%N_type]
                    car_cost = cons + a * max(car_no - m0, 0) ** e
                    reward_sum += (c2 * (1-urgent_prob[(i+a_type)%N_type]) + c3 * urgent_prob[(i+a_type)%N_type]) * (N - car_list_self[i]) + car_cost * car_list_self[i]
                    drive_cost_sum += car_cost * car_list_self[i]
                    transit_cost_sum += (c2 * (1-urgent_prob[(i+a_type)%N_type]) + c3 * urgent_prob[(i+a_type)%N_type]) * (N - car_list_self[i])
                elif N_type == 1:
                    car_no = car_list_full[i]
                    car_cost = cons + a * max(car_no - m0, 0) ** e
                    reward_sum += (c2 * (1-urgent_prob[i]) + c3 * urgent_prob[i]) * (N - car_list_self[i]) + car_cost * car_list_self[i]
                    drive_cost_sum += car_cost * car_list_self[i]
                    transit_cost_sum += (c2 * (1-urgent_prob[i]) + c3 * urgent_prob[i]) * (N - car_list_self[i])
        reward_population_temp = reward_sum / N
        drive_cost_population_temp = drive_cost_sum / N / 4 # drive 4 days in total
        transit_cost_population_temp = transit_cost_sum / N
        if theory == 0: # calculating theory results
            reward_agent_temp = new_Q[0][0][0].max() * (1-urgent_prob[0]) + new_Q[0][0][1][0] * urgent_prob[0]
        else:
            reward_agent_temp = None
    else:
        reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp = 0, 0, 0, 0
    return reward_population_temp, drive_cost_population_temp, transit_cost_population_temp, reward_agent_temp

def get_pi(N_type, reward_population, reward_agent, Q, population_pi):
    #print('reward_population:', reward_population, 'reward_agent:', reward_agent)
    new_pi = {}
    agent_pi = {}
    for a_type in range(N_type):
        reward_population_temp = reward_population[a_type]
        if reward_population_temp == 0: # implies this type has no agents, do not learn
            learning_rate = 0
        else:
            reward_agent_temp = reward_agent[a_type]
            delta_reward = abs(reward_agent_temp - reward_population_temp)
            learning_rate = min(1, beta * delta_reward / abs(reward_population_temp) + 0.001)
        temp_pi = np.zeros([2,5,2,2])
        new_pi_temp = np.zeros([2,5,2,2])
        for i in range(N_bus_already):
            for j in range(N_day):
                for k in range(N_urgent):
                    for m in range(N_action):
                        if i == 0 and j < 4 and k == 0: # only decise in four scenarios
                            action_max = np.where(Q[a_type][i][j][k] == np.max(Q[a_type][i][j][k]))[0]
                            if abs(Q[a_type][i][j][k][0] - Q[a_type][i][j][k][1]) < min_Q_diff: # q diff is smaller than the min_Q_diff, regard them as the same
                                temp_pi[i][j][k][m] = population_pi[a_type][i][j][k][m] #keep the same with existing
                            else:
                                action = action_max[0]
                                temp_pi[i][j][k][action] = 1
        for i in range(N_bus_already):
            for j in range(N_day):
                for k in range(N_urgent):
                    for m in range(N_action):
                        if i == 0 and j < 4 and k == 0: # only decise in four scenarios
                            new_pi_temp[i][j][k][m] = population_pi[a_type][i][j][k][m] * (1 - learning_rate) + temp_pi[i][j][k][m] * learning_rate
        new_pi[a_type] = copy.deepcopy(new_pi_temp)
        agent_pi[a_type] = copy.deepcopy(temp_pi)
    return new_pi, agent_pi

def get_difference(N_type, state_action_distribution, new_state_action_distribution, new_population_pi, agent_pi):
    diff_L = [0]*N_type
    diff_pi = [0]*N_type
    for a_type in range(N_type):
        for i in range(N_bus_already):
            for j in range(N_day):
                for k in range(N_urgent):
                    for m in range(N_action):
                        diff_L[a_type] += abs(new_state_action_distribution[a_type][i][j][k][m] - state_action_distribution[a_type][i][j][k][m]) #** 2
                        diff_pi[a_type] += abs(new_population_pi[a_type][i][j][k][m] - agent_pi[a_type][i][j][k][m]) #** 2
    return diff_L, diff_pi


def get_car_cost_function(cost_function,c2,N_agent):
    if cost_function == 'linear':
        cons = 0
        m0 = 0
        a = c2 / N_agent # cost parameter for car
        e = 1
    elif cost_function == 'quadratic':
        cons = 0
        m0 = 0
        a = c2 / (N_agent * N_agent)
        e = 2
    elif cost_function == 'piecewise':
        cons = 1
        m0 = N_agent/2
        a = (c2 - cons) / (N_agent - m0)
        e = 1
    return cons, m0, a, e
               
def outer_train(N_type,urgent_prob,c3,agent_distribution):  
    population_pi = initialize_pi(N_type, homogenerous_pi0, pc)  # initialize the policy for population
    new_state_distribution, new_state_action_distribution = initialize_state_action_distribution(N_type, agent_distribution, population_pi)
    for epoch in range(epochs):
        Q = {}
        reward_population = {}
        drive_cost_population = {}
        transit_cost_population = {}
        reward_agent = {}
        for a_type in range(N_type):
            reward_population[a_type], drive_cost_population[a_type], transit_cost_population[a_type], reward_agent[a_type], Q[a_type] = Value_iteration(N_type, agent_distribution, Q_epochs,population_pi,epoch,a_type)
        new_population_pi, agent_pi = get_pi(N_type, reward_population, reward_agent, Q, population_pi)
        state_distribution = copy.deepcopy(new_state_distribution)
        state_action_distribution = copy.deepcopy(new_state_action_distribution)
        new_state_distribution, new_state_action_distribution = update_state_action_distribution(N_type, state_distribution,new_population_pi)
        diff_L, diff_pi = get_difference(N_type, state_action_distribution, new_state_action_distribution, new_population_pi, agent_pi)
        population_pi = copy.deepcopy(new_population_pi)
        sum_diff_pi = np.sum(diff_pi)
        phi_list = []
        if N_type == 1:
            phi_list = [population_pi[a_type][0][0][0][1], population_pi[a_type][0][1][0][1], population_pi[a_type][0][2][0][1], population_pi[a_type][0][3][0][1]]    
        else:
            for a_type in range(5):
                phi_list.append([population_pi[a_type][0][0][0][1], population_pi[a_type][0][1][0][1], population_pi[a_type][0][2][0][1], population_pi[a_type][0][3][0][1]])
        if sum_diff_pi < 0.0000001: # if the current policy is almost the same with previous policy, stop iteration
            break
    return reward_population, drive_cost_population, transit_cost_population, population_pi


N_bus_already = 2 # bus_already has two values: 0 means have not taken bus, 1 means have taken bus
N_day = 5 # number of days in a cycle
N_urgent = 2 # urgent has two values: 0 means not urgent, 1 means urgent
N_action = 2 #action has two values: taking bus or driving car

homogenerous_pi0 = 1 # set same pi0 for different agent types if 1, 0 otherwise, setting 1 can facilitate the training especially for N_type = 5
pc = 0 # the prob of choosing car when have choice if setting homogenerous pi0
gamma = 1.0 # discount for future reward
beta = 0.01 # 0.01 # learn rate parameter
truncation = 1 # 1 if trauncate the L distribution, 0 otherwise

Q_epochs = 5000  # max iteration number for Q-table value iteration
epochs = 50000  # max iteration for population convergence
N_agent = 20  # full population
c2 = -6 # bus cost when not urgent
c3 = -6*3 # bus cost when urgent
cost_function = 'linear' # linear, quadratic, piecewise
# car_cost = cons + a * [max(car_no - m0, 0)] ** e
cons, m0, a, e = get_car_cost_function(cost_function,c2,N_agent)

N_type = 1 # 1 means original Flexible Restriction Policy (O-FRP), 5 means Interlocking Flexible Restriction Policy (IL-FRP)

urgent_prob = [0.5]*5 # the uurgency distribution from Monday to Friday
min_Q_diff = 0.001 # when q_vule difference in two iterations are <  min_Q_diff, keep the previous pi, otherwise, update the pi to choose action with larger q_value
for i in range(5):
    min_Q_diff *= urgent_prob[i] # the smaller the urgency probability, the samller the threshold
agent_distribution_equal = 1 # 1 if agents are equally distributed across different groups, 0 otherwise. Takes effect only for IL-FRP
agent_distribution = solve_agent_distribution(N_type, urgent_prob, agent_distribution_equal) # get the agent numbers in each group

# learn the policy pi,the returned population_pi is the learned UE
# total_cost_dic is each group's total cost in UE, 
# drive_cost_dic is each group's average drive cost in UE 
# transit_cost_dic is each group's average transit cost in UE
total_cost_dic, drive_cost_dic, transit_cost_dic, population_pi = outer_train(N_type,urgent_prob,c3,agent_distribution)













