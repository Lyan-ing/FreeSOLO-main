import os
import urllib
import urllib.request
print('\nbegin...')
url = 'https://objects.githubusercontent.com/github-production-release-asset-2e65be/350409920/40b50780-8b37-11eb-9027-6ef5790cdeef?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T095029Z&X-Amz-Expires=300&X-Amz-Signature=d48071e0989925924b61e03567e056679bcd74592a72ed0d1b14ed3ce1621e37&X-Amz-SignedHeaders=host&actor_id=65017880&key_id=0&repo_id=350409920&response-content-disposition=attachment%3B%20filename%3Ddpt_hybrid-midas-501f0c75.pt&response-content-type=application%2Foctet-stream'
path = '/home/yl/python/Self-Super/FreeSOLO-main/training_dir/pre-trained/DenseCL'
urllib.request.urlretrieve(url, path+ 'densecl_r101_imagenet_200ep.pth')
print('done!')


# https://blog.csdn.net/m0_37644085/article/details/81948396
import glob  # 导入模块

path_file_number = glob.glob('/media/data/miki/datasets/COCO/unlabeled2017/*.jpg')  # 指定路径的文件夹里的文件
path_file_number2 = glob.glob('/media/data/miki/datasets/COCO/train2017/*.jpg')  # 指定路径的文件夹里的文件

# file_number = glob.glob(pathname='*.jpg')  # 当前文件夹下的文件

# print(path_file_number)     # 返回一个列表
a = len(path_file_number)
b = len(path_file_number2)
c =a + b
print(c)

save_path='/media/data/miki/datasets/COCO'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# print("\ndownloading with urllib")
# url = 'http://images.cocodataset.org/zips/unlabeled2017.zip'
# urllib.request.urlretrieve(url, save_path+"/unlabeled2017.zip")  # 19GB
# print('done!')

# print ("downloading with urllib")
# # url = 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/test_a.zip'
# # # print("downloading with urllib")
# # urllib.request.urlretrieve(url, save_path+"/test_a.zip")  # 314.49MB
# print('done!')
#
# url = 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/train.zip'
# # print("downloading with urllib")
# urllib.request.urlretrieve(url, save_path+"/train.zip")  # 3.68GB
# print('done!')
#
# url = 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/train_mask.csv.zip'
# # print("downloading with urllib")
# urllib.request.urlretrieve(url, save_path+"/train_mask.csv.zip")
# print('done!')
#
# url = 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/test_a_samplesubmit.csv'
# # print("downloading with urllib")
# urllib.request.urlretrieve(url, save_path+"/test_a_samplesubmit.csv")
# # 地址，文件夹/保存文件名
# # urllib.request.urlretrieve(url_base + file_name, file_path)
# print('done!')

import zipfile
#zipfile解压
# save_path ='/media/data/yl/Dataset/Tianchi/Seg'

# z = zipfile.ZipFile(save_path+"/test_a.zip", 'r')
# z.extractall(path=save_path)
# # 解压缩地址
# print('done!')
# z.close()

z = zipfile.ZipFile('/data8T/miki/COCO/images/unlabeled2017(1).zip', 'r')
z.extractall(path=save_path)
# 解压缩地址
print('done!')
z.close()