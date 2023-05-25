#モジュールインポート
from scipy.interpolate import lagrange
import scipy.interpolate as scipl
import numpy as np
import matplotlib.pyplot as plt

#前提条件
x_lis=[1,2,3,5,10,12,20]
y_lis=[14.1,16.5,22.9,31.3,46,44.8,63.2]

#補間式
x=np.arange(1,21,0.1)
f_lag=lagrange(x_lis,y_lis)  #ラグランジュ補間
f_sci=scipl.CubicSpline(x_lis,y_lis)  #スプライン補間

#グラフ出力
plt.plot(x_lis, y_lis,'o')
plt.plot(x, f_lag(x),'-')
plt.plot(x, f_sci(x),'-')
plt.xlabel('x [-]')
plt.ylabel('y [-]')
plt.legend(['Row data','Lagrange','Cubic Spline'])
plt.show()