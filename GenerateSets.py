from random import randrange

N = 40
SubsetSize = 10
all = list(range(N))
subset = []


def get_rnd(arr: list) -> int:
    rnd = randrange(0, len(arr))
    return arr.pop(rnd)


for _ in range(SubsetSize):
    subset.append(get_rnd(all))


for i in range(30):
    if randrange(0, 4) < 1:
        for _ in range(randrange(0, 4)):
            popped = get_rnd(subset)
            subset.append(get_rnd(all))
            all.append(popped)
    print(sorted(subset), ',', sep='')
