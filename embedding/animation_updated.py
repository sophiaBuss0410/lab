import glob
from PIL import Image
 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## 参考：https://rightcode.co.jp/blog/information-technology/python-artistanimation-creating-animations-multiple-diagrams

#フォルダ名を入れます 
folderName = 'embedding/fig_updated'

#該当フォルダから画像のリストを取得。読み込みたいファイル形式を指定。ここではpng 
picList = glob.glob(folderName + '/*.png')
# print(picList)
     
#figオブジェクトの作成
fig = plt.figure()

#余白をなくす
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

#figオブジェクトから目盛り線などを消す
plt.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)

#空のリスト作成
ims = []
     
#画像ファイルを空のリストの中に1枚ずつ読み込み
for i in range(len(picList)):
         
    #読み込んで付け加えていく
    tmp = Image.open(picList[i])
    ims.append([plt.imshow(tmp)])     
     
#アニメーション作成  
ani = animation.ArtistAnimation(fig, ims, interval=800)

# ffmpeg関連
# 参考：https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest
# 参考：https://novnote.com/matplotlib-animation-ffmpeg/468/#matplotlibrc

#アニメーション保存
ani.save("embedding/all_4dots_jp_updated.mp4", writer="ffmpeg", dpi=300)