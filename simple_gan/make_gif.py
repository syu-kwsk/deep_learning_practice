import os
import glob
from PIL import Image

path_list = sorted(glob.glob(os.path.join(*['save_images', '*'])))

imgs = []                                                   # 画像をappendするための空配列を定義

for i in range(len(path_list)):
    img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
    imgs.append(img) 

imgs[0].save(
    'save_images/result.gif',
    save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0)
