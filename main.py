import numpy as np
from random import randrange
# import matplotlib.pyplot as plt
import io
import json

N: int = 0
M: int = 0
VALUES: np.array = np.zeros(1)

POPULATION_SIZE: int = 40
ITERATION_COUNT: int = 700

MUTATION_RATE: float = 0.12
STOP_POINT: float = 0.15


def check_constrain(solution: np.array) -> bool:
    unique = np.unique(solution)
    return len(unique) == N


# std of sums
def objective_funtion(solution: np.array):
    if not check_constrain(solution):
        return np.inf
    return np.std(get_sums(solution))


# splits solution into teams and calculates their sums
def get_sums(solution: np.array):
    return np.sum([VALUES[s] for s in np.split(solution, M)], 1, dtype=int)


def random_solution():
    rnd = np.array(range(N), dtype=int)
    np.random.shuffle(rnd)
    return rnd


def choose_one(population: list[np.array]):
    i = randrange(0, round(POPULATION_SIZE / 2))
    return population[i]


# Choose random part from sol_a, fill the rest in order of sol_b
def crossover(sol_a: np.array, sol_b: np.array):
    start = randrange(0, N-1)
    end = randrange(start+1, N)
    rnd_a = sol_a[range(start, end)]
    child = []

    for b in sol_b:
        if b not in rnd_a:
            child.append(b)

    child = child[:start] + list(rnd_a) + child[start:]
    # child = list(rnd_a) + child
    return np.array(child, dtype=int)


# swap two values in solution
def mutate(sol: np.array, rate: float):
    for i in range(N):
        if np.random.rand() > rate:
            continue
        rnd = randrange(0, N)
        sol[i], sol[rnd] = sol[rnd], sol[i]
    return sol


def run(values: np.array, team_count: int, ids):
    global VALUES, M, N
    VALUES = values
    M = team_count
    N = len(values)

    bests = []
    fitnesses = []

    population = [
        random_solution() for _ in range(POPULATION_SIZE)
        ]

    iteration = 0
    while iteration < ITERATION_COUNT:
        iteration += 1

        # sort population, get top half
        Q = sorted(range(POPULATION_SIZE),
                   key=lambda x: objective_funtion(population[x]),
                   reverse=False)
        tophalf = [population[q] for q in Q][:round(POPULATION_SIZE/2)]
        new_population = tophalf.copy()
        # new_population = []

        # save best and average objective_funtion value
        best = population[Q[0]]
        th_f = list(map(lambda x: round(objective_funtion(x), 3), tophalf))
        bests.append(objective_funtion(best))
        fitnesses.append(round(sum(th_f) / len(th_f), 3))

        # terminal condition
        # print(np.std(fitnesses[-30:]))
        if np.std(fitnesses[-30:]) < STOP_POINT and iteration > 30:
            break

        # create new population
        while len(new_population) < POPULATION_SIZE:
            a = choose_one(tophalf)
            b = choose_one(tophalf)

            child = crossover(a, b)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)
        population = new_population.copy()

    # print(bests)
    # print(fitnesses)
    # print()

    # print(iteration)
    # print(VALUES)
    Q = sorted(range(POPULATION_SIZE),
               key=lambda x: objective_funtion(population[x]),
               reverse=False)
    best = population[Q[0]]
    sums = get_sums(best)
    # print('best:\n'
    #       f'{[list(s) for s in np.split(best, M)]}')
    print(f'sums: {list(sums)}  '
          f'{max(sums) - min(sums)}\n'
          f'std:  {objective_funtion(best)}')
    print(f'Ids: {[list(s) for s in np.split(ids[best], M)]}')
    print(f'Values: {[list(VALUES[s]) for s in np.split(best, M)]}')
    # print(f'Items: [{", ".join(map(str, sorted(VALUES)))}]')

    # x_axis = list(range(iteration))
    # plt.plot(x_axis, fitnesses, label='avg')
    # plt.plot(x_axis, bests, label='best')
    # plt.xlabel("iteration")
    # plt.ylabel("teams diversity")
    # plt.legend()
    # plt.show()

    data = {
            'sums': f"{list(sums)}",
            'std': f"{round(objective_funtion(best), 3)}",
            'ids': f"{[list(s) for s in np.split(ids[best], M)]}",
            'vals': f"{[list(VALUES[s]) for s in np.split(best, M)]}"
    }

    return data


if __name__ == '__main__':
    print()
    # ranks = np.array([
    #     9, 9, 7, 5,
    #     5, 5, 3, 3,
    #     2, 2, 1, 1,
    #     ])
    ids = np.array([
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22, 24, 27, 32, 34, 38, 39],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22, 24, 27, 32, 34, 38, 39],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22, 24, 27, 32, 34, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 19, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 31, 32, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 21, 22, 23, 24, 27, 31, 32, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        [0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19, 21, 22, 23, 24, 27, 31, 32, 34, 37, 38, 39],
        ])
    ids = np.array([
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 1, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 4, 5, 6, 7, 9, 11, 15, 17, 20, 22, 23, 24, 25, 30, 31, 34, 35, 38, 39],
        [0, 1, 5, 6, 7, 9, 11, 17, 18, 20, 23, 24, 25, 30, 31, 34, 35, 37, 38, 39],
        [0, 1, 5, 6, 7, 9, 11, 17, 18, 20, 23, 24, 25, 30, 31, 34, 35, 37, 38, 39],
        [0, 1, 5, 6, 7, 9, 11, 17, 18, 20, 23, 24, 25, 30, 31, 34, 35, 37, 38, 39],
        [0, 1, 5, 6, 7, 9, 11, 17, 18, 20, 23, 24, 25, 30, 31, 34, 35, 37, 38, 39],
        [0, 1, 5, 6, 7, 9, 11, 17, 18, 20, 23, 24, 25, 30, 31, 34, 35, 37, 38, 39],
        [0, 1, 5, 6, 7, 9, 11, 17, 18, 20, 23, 24, 25, 30, 31, 34, 35, 37, 38, 39],
        ])
    ranks = np.array([
        11, 12, 12, 14, 15, 15,
        23, 25, 27, 30, 35, 35,
        36, 38, 44, 44, 45, 46,
        46, 47, 47, 48, 49, 49,
        51, 51, 55, 59, 61, 62,
        64, 65, 66, 70, 73, 77,
        81, 85, 86, 95
        ], dtype=int)
    team_size = 5

    all_data = []
    for i in ids:
        subset = ranks[i]
        # subset = np.power(subset, 2)
        print()
        data = run(subset, round(len(subset)/team_size), i)
        all_data.append(data)

    with io.open('AI-output.json', 'w', encoding='utf8') as file:
        str_ = json.dumps(all_data,
                          indent=2, sort_keys=False,
                          separators=(',', ':'), ensure_ascii=True)
        file.write(str_)
