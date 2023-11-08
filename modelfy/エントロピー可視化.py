# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
i1 = np.arange(0.05,1,0.05)
i2 = 1-i1
# print(i1, i2, len(i1), 49*49)

U = 637000
M = 9559
G = 3060

pm1 = M/U
pm2 = 1-pm1
pg1 = G/U
pg2 = 1-pg1

# %%
Him = [[],[],[],[],[]]

for u in i1:
    for m in i1:
        pi1m1 = m*pm1
        pi2m1 = (1-m)*pm1
        pi1m2 = u*pm2
        pi2m2 = (1-u)*pm2
        Him[0].append(u)
        Him[1].append(m)
        Him[2].append(-pi1m1*np.log2(m)-pi2m1*np.log2(1-m)-pi1m2*np.log2(u)-pi2m2*np.log2(1-u))
        Him[3].append(-u*np.log2(u)-(1-u)*np.log2(1-u))
        Him[4].append(-m*np.log2(m)-(1-m)*np.log2(1-m))

# %%
df = pd.DataFrame({"P(i1|U)":Him[0], "P(i1|M)":Him[1], "H(I|M)": Him[2], "H(I)":Him[3], "H(M)":Him[4]})

# %%
x1 = df[["P(i1|U)"]]
y1 = df[["P(i1|M)"]]
z1 = df[["H(I|M)"]]

plt.figure(figsize=(6, 5))
axes = plt.axes(projection="3d")
print(type(axes))
axes.scatter3D(x1, y1, z1)
x2=0.456
y2=0.223
z2=0.9909732

# x3=0.456
# y3=0.198
# z3=0.993078427

axes.scatter3D(x2, y2, z2, s=70, c='orange')
# axes.scatter3D(x3, y3, z3, s=70, c='green')

# axes.set_zlim(0.9,1)

axes.set_xlabel("P(i1|U)")
axes.set_ylabel("P(i1|M)")
axes.set_zlabel("H(I|M)")
plt.show()
