import numpy as np
arr = np.random.randint(1, 51, (5, 4))
print("Array:\n", arr)
anti_diag = [arr[i, -i - 1] for i in range(min(arr.shape))]
print("Anti-diagonal:", anti_diag)
row_max = np.max(arr, axis=1)
print("Max in each row:", row_max)
mean_val = arr.mean()
filtered = arr[arr <= mean_val]
print("Elements <= mean:", filtered)

def numpy_boundary_traversal(matrix):
    result = []
    result.extend(matrix[0])
    result.extend(matrix[1:-1, -1])
    result.extend(matrix[-1][::-1])
    result.extend(matrix[1:-1, 0][::-1])
    return list(result)
