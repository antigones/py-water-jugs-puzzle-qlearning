from water_in_jugs_qlearning import WaterInJugsQLearning, Jug
import itertools



def main():

    jugs = [Jug('Jug1', 8, 8), Jug('Jug2', 5, 0), Jug('Jug3', 3, 0)]
    actions = itertools.permutations(range(len(jugs)), 2)
    wcg_arena = WaterInJugsQLearning(
        start_state=jugs,
        actions=actions,
        goal_qty=4,
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = wcg_arena.train()

    print('*** SOLUTION ***')
    for step in solution_steps:
        print(step)

    print('*** SCORES ***')
    for score, e in zip(scores, eps_list):
        print("{score:.2f};{epsilon:.2f}".format(score=score, epsilon=e))
    

if __name__ == "__main__":
    main()