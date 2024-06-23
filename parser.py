import json
import io
from itertools import combinations

with io.open('AI-output.json') as file:
    data = json.load(file)

LINES = []


def printf(*strings,
           sep: str = ' ', end: str = '\n'):
    line = sep.join(strings)
    LINES.append(line + end)


all_sums = []
all_stds = []
all_ids = []
for d in data:
    all_stds.append(float(d['std']))

    ids = eval(d['ids'])
    for j, tt in enumerate(ids):
        ids[j] = sorted(tt)
    all_ids.extend(ids)

    sums = eval(d['sums'])
    all_sums.append(sums)


# print(f'avg_std: {round(avg_std / len(data), 3)}')
# print(*sorted(all_ids), sep='\n')

def count_in(arr, pattern) -> int:
    for p in pattern:
        if p not in arr:
            return 0
    return 1


def flatten(arr):
    return [
        x
        for xs in arr
        for x in xs
        ]


perms2 = combinations(range(40), 2)
perms3 = combinations(range(40), 3)
results2 = []
results3 = []
totals2 = 0
totals3 = 0
for p in perms2:
    count = 0
    for i in all_ids:
        count += count_in(i, p)

    if count > 6:
        results2.append((p, count))
        totals2 += count

for p in perms3:
    count = 0
    for i in all_ids:
        count += count_in(i, p)

    if count > 4:
        results3.append((p, count))
        totals3 += count

flat_ids = flatten(all_ids)
# print(sorted(flat_ids))
results2 = sorted(results2, key=lambda x: x[1])
for pair, count in results2:
    printf(f'{pair}: {count}x  {[flat_ids.count(p) for p in pair]}')
printf(f'total: {totals2}')
printf()

results3 = sorted(results3, key=lambda x: x[1])
for trip, count in results3:
    printf(f'{trip}: {count}x  {[flat_ids.count(t) for t in trip]}')
printf(f'total: {totals3}')
printf()

printf(f'Avg std: {sum(all_stds) / len(all_stds)}')
worst_std = max(all_stds)
worst_sums = all_sums[all_stds.index(worst_std)]
printf(f'Worst std: {worst_std}')
printf(f'Worst sums: {worst_sums}')
printf(f'Max diff: {max(worst_sums)-min(worst_sums)}')

print(all_stds)

with open('AI-pairs.txt', 'w', encoding='utf8') as file:
    for line in LINES:
        file.write(line)
