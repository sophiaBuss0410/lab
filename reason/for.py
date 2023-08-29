import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

from sentence_transformers import SentenceTransformer
from scipy.interpolate import griddata
import matplotlib.cm as cm

op = [
    "Because I think it is better for the children's growth, etc., if the wife takes care of the home.",
    "Because I think it is difficult for a wife to continue working while balancing housework, childcare, and elder care",
    "Because I think it is better for the husband to work outside the home to earn more money",
    "Because I think it is the traditional Japanese way of family",
    "Because my parents divided roles in the family.",
    ]

model = SentenceTransformer('all-MiniLM-L6-v2')

gend = model.encode(op)

weights = gend
vec = weights.tolist()

df1 = pd.DataFrame(op, columns=['sentence'])
df = pd.DataFrame((vec[i] for i in range(len(op))), index=df1)


# Computing the correlation matrix
X_corr = df.corr()
# Computing eigen values and eigen vectors
values, vectors = np.linalg.eig(X_corr)
# Sorting the eigen vectors coresponding to eigen values in descending order
args = (-values).argsort()
values = vectors[args]
vectors = vectors[:, args]
# Taking first 2 components which explain maximum variance for projecting
new_vectors = vectors[:,:2]

# Projecting it onto new dimesion with 2 axis
neww_X = np.dot(vec, new_vectors)
# 虚部の削除
neww_X = np.real(neww_X)
neww_X

# Importing survey data
data = pd.read_csv('reason\賛成_2014_2019_es.csv', encoding="shift-jis")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(16,12), dpi=330, subplot_kw={'projection': '3d'})

# Initializing index
h = 0

for i in range(2):
    for j in range(2):

        axes[i][j].scatter(neww_X[:,0], neww_X[:,1], [0 for i in range(len(op))], linewidths=1,color='yellow')
        # vocab=op
        jp = [
            "妻が家庭を守った方が、子供の成長などにとって良いと思うから",
            "家事・育児・介護と両立しながら、妻が働き続けることは大変だと思うから",
            "夫が外で働いた方が、多くの収入を得られると思うから",
            "日本の伝統的な家族の在り方だと思うから",
            "自分の両親も役割分担をしていたから"]
        for k, word in enumerate(jp):
            axes[i][j].text(neww_X[k,0]+0.01, neww_X[k,1]+0.01, 0, word, zdir=None)

        # calculate potential
        year1979 = pd.DataFrame({data.iloc[h, 0] : data.iloc[h, 1:]})
        year1979['pot'] = -np.log(year1979.iloc[:, 0].astype(np.float64)/100)

        # bar graph code
        x, y, z = neww_X[:,0], neww_X[:,1], year1979.iloc[:, 0].astype(np.float64).tolist()

        top = z
        bottom = np.zeros_like(top)
        width = 0.01
        depth = 0.01
        axes[i][j].bar3d(x, y, bottom, width, depth, top, shade=True)
        axes[i][j].set_title(data.iloc[h, 0])#, size=20)
        h += 1

        ## surface plot code
        # border condition
        # dum = [-0.4, 0.4, -0.4, 0.4]
        # dumy = [-0.4, 0.4, 0.4, -0.4]
        
        dum = [-0.5, 0.5, -0.5, 0.5]
        dumy = [-0.5, 0.5, 0.5, -0.5]


        xx = neww_X[:,0].tolist()
        yy = neww_X[:,1].tolist()

        for l in range(len(dum)):
            xx.append(dum[l]) 
            yy.append(dumy[l])

        zz = 10*year1979.iloc[:, 1]
        zz = zz.to_list()
        for m in range(len(dum)):
            zz.append(-10*np.log(1/100))
        
        X, Y, Z = xx, yy, zz
        print(Z)

        points = np.array([X, Y]).T

        # create a grid of coordinates between the minimum and
        # maximum of your X and Y. 50j indicates 50 discretization
        # points between the minimum and maximum.
        X_grid, Y_grid = np.mgrid[-0.5:0.5:100j, -0.5:0.5:100j]
        # interpolate your values on the grid defined above
        Z_grid = griddata(points, Z, (X_grid, Y_grid), method='cubic')
        idx = np.unravel_index(np.argmin(Z_grid), Z_grid.shape)
        print(idx)

        axes[i][j].contour3D(X_grid, Y_grid, Z_grid, levels=20, cmap=cm.coolwarm)

        # axes[i][j].set_zlim(0, 50)
        # axes[i][j].set_xticks([-0.2, 0, 0.2])
        # axes[i][j].set_yticks([-0.2, 0, 0.2])

        # lowest point
        x = -0.5 + 1/100*idx[0]
        y = -0.5 + 1/100*idx[1]
        axes[i][j].scatter(x, y, -40)

        axes[i][j].view_init(elev=50, azim=290, roll=0)

        # axes[i][j].set_xlabel("主成分1", size=12)
        # axes[i][j].set_ylabel("主成分2", size=12)
        # axes[i][j].set_zlabel("\n回答者の割合 (%), \nポテンシャル*10", size=15)

        if h == 3:
            break

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('reason/contour_for_es.png', bbox_inches='tight', pad_inches=0.3)
# plt.show()

