import math

# euclidean distance between two vector (one dimensional array)
def euclidean_distance(vec1, vec2) -> float:
    dist = 0.0
    for i in range(len(vec1) - 1):
        dist += (vec1[i] - vec2[i])**2
    return math.sqrt(dist)