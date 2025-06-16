import numpy as np
arr = np.random.uniform(0, 10, 20)
arr = np.round(arr, 2)
print("Array:", arr)
print("Min:", np.min(arr))
print("Max:", np.max(arr))
print("Median:", np.median(arr))
arr[arr < 5] = np.square(arr[arr < 5])
print("Transformed:", arr)

def numpy_alternate_sort(array):
    array = np.sort(array)
    result = []
    i, j = 0, len(array) - 1
    while i <= j:
        result.append(array[i])
        if i != j:
            result.append(array[j])
        i += 1
        j -= 1
    return np.array(result)
