# -*- coding:utf-8 -*- 
__author__ = 'xuy'

'''
主要参考wider_annotation.py文件，验证集val-dataset将xmin,ymin,xmax,ymax的文件输出到mAP的gt文件夹下
生成的这个文件可以用来给draw_gt.py提供数据
具体内容格式参考mAP文件夹下的my_ground-truth文件
我们可以根据由wider_annotation.py生成的WIDER_train_val.txt[一共3226张图片]文件来进行更改
'''
import os

input_file='/home/xuy/code/keras-yolo3-detection/wider_dataset/WIDER_val.txt'
output_path='/home/xuy/code/mAP/ground-truth/'


def read_file(data_file, mode='more'):
    """
    读文件, 原文件和数据文件
    :return: 单行或数组
    """
    try:
        with open(data_file, 'r') as f:
            if mode == 'one':#只有一个候选框
                output = f.read()
                return output
            elif mode == 'more':#有多个候选框，因此需要readlines
                output = f.readlines()
                # return map(str.strip, output)
                return output
            else:
                return list()
    except IOError:
        return list()

data_lines=read_file(input_file)
for data_line in data_lines:
    data_line=data_line.split()#data_line是每一组的信息，data_line[0]是路径，data_line[1...-1]是5位数组结果
    pic_filename=os.path.basename(data_line[0])
    # print(data_line[-1])
    portion = os.path.splitext(pic_filename)
    if portion[1] == '.jpg':
        txt_result = output_path + portion[0] + '.txt'
    # print('txt_result的路径是：' + txt_result)#每个文件的txt输出路径
    #需要遍历data_line【1～  -1】
    # for data in range(1,len(data_line)):
    #     print(data[])
    for i in range(1,len(data_line)):
        xmin,ymin,xmax,ymax,class_label=data_line[i].split(',')
        print(xmin,ymin,xmax,ymax)
        with open(txt_result, 'a')as new_f:
            new_f.write('face'+' '+xmin+' '+ymin+' '+xmax+' '+ymax+'\n')




