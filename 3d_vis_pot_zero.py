import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
import matplotlib.cm as cm


file = 'id909'
json_open = open(file + '.json', 'r', encoding="utf-8")
json_load = json.load(json_open)
# print(json_load)
d = json_load
# print(d.keys())

# ノード情報
nodes = d["data"]["items"]["artifacts"]

# エッジ情報
arr = d["data"]["arrows"]["hasArtifactsOnly"]

lst = [[nodes[i]['id'], nodes[i]['fT'], nodes[i]['sT'], nodes[i]['pos']['y'], nodes[i]['pos']['x']] for i in range(len(nodes))]
df = pd.DataFrame(lst, columns =['id', 'fT', 'sT', 'x', 'y']) 
# df.head()

# 仮想的な指標を付与
num = np.random.randint(1, 100, len(df))
df["num"] = num

# ポテンシャルの計算
N = df["num"].sum()
df["pot"] = -np.log(df["num"]/N)

ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = df["x"], df["y"], df["pot"]

points = np.array([X, Y]).T
# print(points)

# create a grid of coordinates between the minimum and
# maximum of your X and Y. 50j indicates 50 discretization
# points between the minimum and maximum.
X_grid, Y_grid = np.mgrid[1:1500:100j, 1:1500:100j]
# interpolate your values on the grid defined above
Z_grid = griddata(points, Z, (X_grid, Y_grid), method='cubic')

Z_d = np.nan_to_num(Z_grid)

# print(Z_d[0])

ax.plot_surface(X_grid, Y_grid, Z_d, cmap=cm.coolwarm, 
                       linewidth=1, antialiased=True)

ax.view_init(elev=40, azim=20, roll=0)

plt.savefig("pot_%s.png" % file)
plt.show()