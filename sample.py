# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
import matplotlib.cm as cm

file = 'sample'
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

# %%
num = np.random.randint(1, 10, 12)
# print(num)
df["num"] = num
# print(df)

tf_r = df["fT"].str.contains("RS").to_list()
# tf_b = df["fT"].str.contains("BS").to_list()

for i in range(len(df)):
    if tf_r[i]:
        df.iloc[i, 5] = np.random.randint(6, 10)
    else:
        df.iloc[i, 5] = np.random.randint(1, 6)

# %%
N = df["num"].sum()
df["pot"] = -np.log(df["num"]/N)
# df

# %%
rels = []

for i in range(len(arr)):
  rels.append([arr[i]["i1"], arr[i]["i2"]])

# %%
# グラフの作成

G = nx.Graph()
G.add_edges_from(rels)

pos = {df["id"][i] : np.array([df["x"][i], df["y"][i], 0], dtype=np.float32) for i in range(len(df))}
# print(pos)
pos_ary = np.array([pos[n] for n in G])
# print(pos_ary)

# set up the figure and axes
fig = plt.figure(figsize=(10, 10))
ax2 = fig.add_subplot(111, projection='3d')

ax2.scatter(
    pos_ary[:, 0],
    pos_ary[:, 1],
    pos_ary[:, 2],
    s=200,
)

# ノードにラベルを表示する
# for n in G.nodes:
#     ax2.text(*pos[n], n)

# エッジの表示
for e in G.edges:
    node0_pos = pos[e[0]]
    node1_pos = pos[e[1]]
    xx = [node0_pos[0], node1_pos[0]]
    yy = [node0_pos[1], node1_pos[1]]
    zz = [node0_pos[2], node1_pos[2]]
    ax2.plot(xx, yy, zz, c="#aaaaaa")

# bar graph code
x, y, z = df["x"], df["y"], df["num"]
# print(x, y, z)
top = z
bottom = np.zeros_like(top)
width = 20
depth = 20
ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
# ax2.set_title('支持者数')

# surface plot code
X, Y, Z = df["x"], df["y"], df["pot"]

points = np.array([X, Y]).T

# create a grid of coordinates between the minimum and
# maximum of your X and Y. 50j indicates 50 discretization
# points between the minimum and maximum.
X_grid, Y_grid = np.mgrid[1:1000:100j, 1:1500:100j]
# interpolate your values on the grid defined above
Z_grid = griddata(points, Z, (X_grid, Y_grid), method='cubic')

ax2.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.coolwarm, 
                       linewidth=0, antialiased=True)

ax2.view_init(elev= 25, azim=40, roll=0)

ax2.set_xlabel('Level')
# ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z')
plt.savefig("pot_network_%s.png" % file)

plt.show()
