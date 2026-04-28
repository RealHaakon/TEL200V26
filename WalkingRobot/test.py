import numpy as np

a = np.array([
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
])

b = (a==1)
for row in b:
    print(row)

row, column = np.where(b == False)
indexes = list(zip(row, column))

print(indexes)
