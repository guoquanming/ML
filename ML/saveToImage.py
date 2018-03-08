########################### 特征输出为图片的方法###################################
import random
from PIL import Image


# 获取输出3*64*112*112大小的tensor
def saveToImage(tensor):
# 第一层循环获得64*112*112的tensor
    for i in tensor:
        # 第二层循环获得112*112大小的tensor，即为图片本身
        for a in i:
            # tensor转化为numpy类型
            b = a.data.cpu().numpy()
            # data = np.matrix(b)
            # numpy转图片对象
            new_im = Image.fromarray(b)
            # 设定输出类型为rgb，这一步必须有，是对应图片读取时读取为F类型
            im = new_im.convert('RGB')
            # 随机命名图片+保存输出
            x = random.random()
            im.save('img/' + str(x) + '.jpg', 'jpeg')

##############################################################################