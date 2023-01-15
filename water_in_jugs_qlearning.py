import copy
import random as rd
import numpy as np
from collections import defaultdict



class RLKey:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __hash__(self):
        return hash((tuple(self.k1),tuple(self.k2)))

    def __eq__(self, other):
        return (self.k1, self.k2) == (other.k1, other.k2)

    def __str__(self):
        return str(self.k1)+'|'+str(self.k2)



class WaterInJugsQLearning:

    def __init__(self, start_state, goal_qty, actions, gamma=0.8, max_episodes=50000,
                 epsilon_greedy=True, min_epsilon=0.1, max_epsilon=1.0):
        self.start_state = start_state
        self.goal_qty = goal_qty

        self.gamma = gamma
        self.max_episodes = max_episodes

        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = 0.02
        self.epsilon = self.max_epsilon
        self.epsilon_greedy = epsilon_greedy
        
        self.actions = list(actions)
        

    def get_next_states(self, starting_state):
        next_states = set()
        for action in self.actions:
            next_state = copy.deepcopy(starting_state)

            idx_jug_from = action[0]
            idx_jug_to = action[1]
            
            jug_from = next_state[idx_jug_from]
            jug_to = next_state[idx_jug_to]
            jug_from.pour(jug_to)
            next_states.add(tuple(next_state))

        return next_states

    def get_reward(self, state):
        if self.goal_condition(state):
            return 10
        return -1

    def goal_condition(self, state):
        for jug in state[1:]:
            if jug.qty == self.goal_qty:
                return True
        return False

    def check_qty(self, state):
        c = 0
        for jug in state:
            c+= jug.qty
        if c != 8:
            raise Exception('something went wrong')

    def train(self):

        convergence_count = 0
        q_s_a = defaultdict(int)
        q_s_a_prec = copy.deepcopy(q_s_a)
        episode = 1
        scores = []
        eps_list = []
        rewards = {}
        possible_starting_states = set()
        possible_starting_states.add(tuple(self.start_state))
        for episode in range(1,self.max_episodes):
            initial_state_for_this_episode = self.start_state
            score_per_episode = 0
            
            print("*** EPISODE {episode} ***".format(episode=episode))
            while not self.goal_condition(initial_state_for_this_episode):
                next_states_for_action = self.get_next_states(initial_state_for_this_episode)
                for next_state_for_action in next_states_for_action:
                    self.check_qty(next_state_for_action)
                    possible_starting_states.add(tuple(next_state_for_action))
                    k = RLKey(initial_state_for_this_episode, next_state_for_action)
                    rewards[k] = self.get_reward(next_state_for_action)
                chosen_next_state = rd.choice(tuple(next_states_for_action))

                if self.epsilon_greedy:
                    e = rd.uniform(0, 1)

                    if e > self.epsilon:
                        # action with max value from current state
                        # it's ok to randomly choose if every q is 0 because we would max on a full-0 list
                        s_a_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == initial_state_for_this_episode}
                        if len(s_a_list):
                            m = max(s_a_list, key=s_a_list.get)
                            chosen_next_state = m.k2

                q_s1_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == chosen_next_state}

                m_q_s1 = max(q_s1_list.values(), default=0)

                k_qsa = RLKey(initial_state_for_this_episode,chosen_next_state)
                q_s_a[k_qsa] = rewards[k_qsa] + (self.gamma * m_q_s1)
                score_per_episode += q_s_a[k_qsa]
                initial_state_for_this_episode = chosen_next_state

            if q_s_a == q_s_a_prec:
                if convergence_count > 10:
                    print('** CONVERGED **')
                    break
                else:
                    convergence_count += 1
            else:
                q_s_a_prec = copy.deepcopy(q_s_a)
                convergence_count = 0

            # epsilon update
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

            scores.append(score_per_episode)
            eps_list.append(self.epsilon)
           

        solution_steps = [tuple(self.start_state)]
        next_state = self.start_state
        while not self.goal_condition(next_state):
            candidate_next_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == next_state}
            m = max(candidate_next_list, key=candidate_next_list.get)
            next_state = m.k2
            solution_steps.append(next_state)
        return solution_steps, scores, eps_list

class Jug:

    def __init__(self, name, capacity=0, qty=0):
        self.name = name
        self.capacity = capacity
        self.qty = qty
        self._current_index = 0

    def pour(self, toJug):
        toJug_remaining = toJug.capacity - toJug.qty
        if  toJug_remaining >= self.qty:
            # toJug will not overflow
            toJug.receive(self.qty)
            self.qty = 0
        else:
            # only pour what you can
            self.qty = (toJug.qty + self.qty) - toJug.capacity
            toJug.receive(toJug_remaining)
            
        
    def receive(self, qty):
        self.qty += qty


    def __iter__(self):
        return self

    def __next__(self):
        return self.qty

    def __eq__(self, other):
        return self.name == other.name and self.qty == other.qty

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "{name}: {quantity}".format(name=self.name, quantity=self.qty)

    def __repr__(self):
        return "{name}: {quantity}".format(name=self.name, quantity=self.qty)