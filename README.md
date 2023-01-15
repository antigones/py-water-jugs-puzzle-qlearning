# py-water-jugs-qlearning

Water Jugs is a well-known puzzle where the player has 3 jugs, with the followin initial configuration:

- 8 lt jug (full)
- 5 lt jug (empty)
- 3 lt jug (empty)

Player must obtain 4 lt in any jug by pouring water from a jug to another.

"WaterInJugsQLearning" class uses determistic MDP formula to compute Q (and the optimal policy).

Example usage:

    jugs = [Jug('Jug1', 8, 8), Jug('Jug2', 5, 0), Jug('Jug3', 3, 0)]
    actions = itertools.permutations(range(len(jugs)), 2)
    wj_arena = WaterInJugsQLearning(
        start_state=jugs,
        actions=actions,
        goal_qty=4,
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = wj_arena.train()

Example solution:

    *** SOLUTION ***
    (Jug1: 8, Jug2: 0, Jug3: 0)
    (Jug1: 3, Jug2: 5, Jug3: 0)
    (Jug1: 3, Jug2: 2, Jug3: 3)
    (Jug1: 6, Jug2: 2, Jug3: 0)
    (Jug1: 6, Jug2: 0, Jug3: 2)
    (Jug1: 1, Jug2: 5, Jug3: 2)
    (Jug1: 1, Jug2: 4, Jug3: 3)

Parameters:

- start_state=`[Jug('Jug1', 8, 8), Jug('Jug2', 5, 0), Jug('Jug3', 3, 0)]`
- actions=itertools.permutations(range(len(jugs)), 2)
- goal_qty=4
- gamma=0.8
- max_episodes=1000
- epsilon_greedy=True (specifies if you want to apply an e-greedy policy to compute Q)
    - min_epsilon=0.1
    - max_epsilon=1.0