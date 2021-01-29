from collections import Counter
a = [1, 1, 2, 1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 4, 1, 2, 1,
     2, 1, 2, 3, 2, 1, 2, 3, 4, 1, 2, 1, 3, 2, 2, 1, 2, 1]

for item, freq in Counter(a).items():
    print(freq, round(freq / len(a), 3), round(freq / len(a) * 100, 3))
