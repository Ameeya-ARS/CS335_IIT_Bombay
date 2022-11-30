import json
import numpy as np

def load_points_from_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return np.array(data)

def store_points_to_json(data, filepath):
    assert(len(data.shape) == 2)
    N = data.shape[0]
    points = [ data[i,:].tolist() for i in range(N) ]
    with open(filepath, 'w') as file:
        json.dump(points, file)

def store_labels_to_json(k, labels, filepath):
    next = 0
    label_map = [-1]*k
    remapped_labels = []
    for label in labels:
        assert(label >= 0 and label < k)
        if label_map[label] == -1:
            label_map[label] = next
            next += 1
        remapped_labels += [label_map[label]]
    with open(filepath, 'w') as file:
        data = {}
        data['k'] = k
        data['labels'] = remapped_labels
        json.dump(data, file)