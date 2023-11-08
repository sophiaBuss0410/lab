
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import mlxtend

# %% [markdown]
# ## 偏差値とエントロピーの比較

# %%
df = pd.read_csv("modelfy/data/R5_都立高校_普通科_エントロピー_偏差値_1025.csv")
df = df.iloc[:, :4]

H_I = 0.998417541

xx = np.arange(0.9982, 0.9985, 0.000001)
yy = np.zeros(301) + H_I
zz = np.zeros(301) + 1

# set up the figure and axes
fig = plt.figure(figsize=(8, 8))
ax2 = fig.add_subplot(111, projection='3d')

x, y, z = df["H(I|M)"], df["H(I|G)"], df["偏差値"]-30
# print(x, y, z)

top = z
bottom = np.zeros_like(top)
width = 0.000002
depth = 0.0000001

ax2.bar3d(x, y, bottom, width, depth, top, shade=True)
top2 = zz
bottom2 = np.zeros_like(top2)
ax2.bar3d(xx, yy, bottom2, width, depth, top2, shade=False)

ax2.view_init(elev=40, azim=20) #, roll=0)
ax2.set_xlabel('H(I|M)')
ax2.set_ylabel('H(I|G)')
ax2.set_zlabel('偏差値')

plt.show()


