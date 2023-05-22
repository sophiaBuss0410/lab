import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

from sentence_transformers import SentenceTransformer
from scipy.interpolate import griddata
import matplotlib.cm as cm

op = ["I agree with the idea that 'husbands should work outside the home and wives should take care of the home'.", 
"I rather agree with the idea that 'husbands should work outside the home and wives should take care of the home'.", 
"I rather disagree with the idea that 'husbands should work outside the home and wives should take care of the home'.", 
"I disagree with the idea that 'husbands should work outside the home and wives should take care of the home'."]

model = SentenceTransformer('all-MiniLM-L6-v2')

gend = model.encode(op)

weights = gend
vec = weights.tolist()

df1 = pd.DataFrame(op, columns=['sentence'])
df = pd.DataFrame((vec[i] for i in range(0, 4)), index=df1)


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

# Importing survey data
data = pd.read_excel('embedding\domestic.xlsx')

# Plotting
fig, axes = plt.subplots(3, 4, figsize=(15,15), subplot_kw={'projection': '3d'})

# Initializing index
h = 0

for i in range(3):
    for j in range(4):

        axes[i][j].scatter(neww_X[:,0], neww_X[:,1], [0,0,0,0], linewidths=1,color='blue')
        vocab=op
        for k, word in enumerate(vocab):
            axes[i][j].text(neww_X[k,0], neww_X[k,1], 0, word[:10], zdir=None)

        # calculate potential
        year1979 = pd.DataFrame({data.iloc[h, 0] : data.iloc[h, 1:]})
        year1979['pot'] = -np.log(year1979.iloc[:, 0].astype(np.float64)/100)

        # bar graph code
        x, y, z = neww_X[:,0], neww_X[:,1], year1979.iloc[:-1, 0].astype(np.float64).tolist()
        
        top = z
        bottom = np.zeros_like(top)
        width = 0.01
        depth = 0.01
        axes[i][j].bar3d(x, y, bottom, width, depth, top, shade=True)
        axes[i][j].set_title(data.iloc[h, 0])
        h += 1

        ## surface plot code
        # border condition
        dum = [-0.2, -0.1, 0, 0.1, 0.2, -0.2, -0.1, 0, 0.1, 0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2]
        dumy = [0.2, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1, 0, 0.1, -0.1, 0, 0.1]

        xx = neww_X[:,0].tolist()
        yy = neww_X[:,1].tolist()

        for l in range(len(dum)):
            xx.append(dum[l]) 
            yy.append(dumy[l])

        zz = 10*year1979.iloc[:-1, 1]
        zz = zz.to_list()
        for m in range(len(dum)):
            zz.append(-np.log(1/100))
        
        X, Y, Z = xx, yy, zz

        points = np.array([X, Y]).T

        # create a grid of coordinates between the minimum and
        # maximum of your X and Y. 50j indicates 50 discretization
        # points between the minimum and maximum.
        X_grid, Y_grid = np.mgrid[-0.2:0.2:50j, -0.2:0.2:50j]
        # interpolate your values on the grid defined above
        Z_grid = griddata(points, Z, (X_grid, Y_grid), method='cubic')

        axes[i][j].plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.coolwarm, 
                            linewidth=0, antialiased=True, alpha=.7)

        # axes[i][j].set_ylim(-0.5, 0.5)
        axes[i][j].set_zlim(0, 50)

        axes[i][j].view_init(elev= 25, azim=280, roll=0)

        axes[i][j].set_xlabel("PC1")
        axes[i][j].set_ylabel("PC2")
        axes[i][j].set_zlabel("Z (%)")

        if h == 11:
            break

plt.savefig('embedding/fig/all.png')
plt.show()

