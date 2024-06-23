from random import random
from math import floor

playerCount = 40
values = []

RANKS = [
    # Chall, GM, Master
    0.02, 0.0047, 0.73,
    # Diamond
    0.48, 0.71, 0.83, 2.2,
    # Emerald
    2.6, 2.0, 3.0, 6.4,
    # Plat
    2.7, 3.8, 4.5, 7.1,
    # Gold
    3.4, 4.2, 4.4, 6.3,
    # Silver
    3.1, 3.9, 4.3, 5.5,
    # Bronze
    3.5, 4.3, 4.6, 5.4,
    # Iron
    3.0, 2.5, 1.8, 1.5,
    ]

while len(values) < playerCount:
    rnd, cum = random(), 0
    for i, rank in enumerate(RANKS):
        cum += rank
        if rnd < cum:
            val = 0
            if i < 3:
                val = (i+1) * 2
            else:
                dec = floor((i-3)/4)
                val = dec*10 + ((i-3) % 4) * 2 + 1
            values.append(val)

print(sorted(values))
