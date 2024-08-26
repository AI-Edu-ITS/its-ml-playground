from src.utils import euclidean_distance

# function for get location of similar neighbors
def get_neighbors(train_data, vec_test, num_neighbors: int) -> list:
    dist = []
    neighbors = []
    for vec_train in train_data:
        temp_dist = euclidean_distance(vec_train, vec_test)
        dist.append((vec_train, temp_dist))
    dist.sort(key= lambda tup: tup[1])
    for i in range(num_neighbors):
        neighbors.append(dist[i][0])
    return neighbors

# implement knn
def knn(train_data, test_data, num_neighbors: int) -> list:
    # get list predictions
    preds = []
    for vec_test in test_data:
        neighbors = get_neighbors(train_data, vec_test, num_neighbors)
        output = [row[-1] for row in neighbors]
        temp_preds = max(set(output), key=output.count)
    preds.append(temp_preds)
    return preds