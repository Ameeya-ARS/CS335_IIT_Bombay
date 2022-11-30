import json
from matplotlib import pyplot as plt
from utils import load_points_from_json

points = load_points_from_json('points_2D.json')

X = points[:,0]
Y = points[:,1]

plt.scatter(X,Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('scatter_2D.jpeg')