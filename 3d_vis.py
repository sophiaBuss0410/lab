import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import japanize_matplotlib

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

lst = [[nodes[i]['id'], nodes[i]['fT'], nodes[i]['pos']['y'], nodes[i]['pos']['x']] for i in range(len(nodes))]
df = pd.DataFrame(lst, columns =['id', 'fT',  'x', 'y']) #, dtype = float) 
# df.head()

# 仮想的な指標を付与
num = np.random.randint(1, 10, len(df))
df["num"] = num

# グラフの作成
rels = []

for i in range(len(arr)):
  rels.append([arr[i]["i1"], arr[i]["i2"]])

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
ax2.view_init(elev= 25, azim=15, roll=0)

ax2.set_xlabel('Level')
# ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z')
plt.savefig("3d_vis%s.png" % file)
# plt.show()