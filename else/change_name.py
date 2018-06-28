#coding:utf-8
#用于支持中文
import os

def renameFiles(cur_dir):   
    #列出某个目录下的文件和文件夹，可以是绝对和相对目录  
    files=os.listdir(cur_dir)

    #切换到这个路径作为工作目录,这句要放在listdir后面
    os.chdir(cur_dir) 

    #递归遍历所有文件和文件夹，修改文件名，可以只针对特定后缀的文件更改
    for fileName in files:  
        newName = fileName.replace(' ','_')
        os.rename(os.path.join(cur_dir,fileName),os.path.join(cur_dir,newName))
#        print (fileName)
#    #递归子文件夹
#    for fileName in files:
#        if os.path.isdir(fileName): 
#            print "***scan sub folder***"       
#            renameFiles(fileName)
#            os.chdir(os.pardir) #别忘了切换到父目录
#
#    #对这个目录的文件重命名
#    for i in range(0,len(files)):
#        fileNameArray=os.path.splitext(files[i])
#        if len(fileNameArray)==2 and (fileNameArray[1]==".fbx" or fileNameArray[1]==".FBX"):
#            newFileName=str(i)+fileNameArray[0]+".obj"
#
#            os.rename(files[i],newFileName)
#            print files[i]+" rename file succeeded"

if __name__ == '__main__':
    renameFiles("F:/new_data/to_augmentation/aug_Dog")# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

