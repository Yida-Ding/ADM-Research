import heapq
import numpy as np

a = np.array([1, 3, 2, 4, 5])
b = np.array([5,6,7,8,9])
ind = heapq.nlargest(3, range(len(a)), a.take)
res = b[ind]

print(res)






