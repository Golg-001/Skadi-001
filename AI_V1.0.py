import pygame, sys, time, random
from pygame.locals import *

import win32api, win32con, win32gui
from PIL import ImageGrab

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import math
from PIL import Image as im
import numpy as np
import os
from pynput import keyboard,mouse
import threading
from threading import Lock,Thread
import pyautogui as pag
from matplotlib import pyplot as plt

clock = pygame.time.Clock() #使用pygame的time.Clock()控制最大的每秒运行数量

#变量初始化
press_key,esc,print_key = 0,0,0
x1,y1 = 0,0

################
#全为按键监听函数
def on_click(x, y, button, pressed):
    global x1,y1
    if pressed:
        x1,y1 = x,y
        return False
def click_pos():
    global x1,y1
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    return x1,y1
def on_press(key):
    """按下按键时执行。"""
    global press_key,esc,print_key
    if key is keyboard.Key.space:
        press_key = 0.5
    if key is keyboard.Key.esc:
        esc = 1
        return False
    if key is keyboard.Key.enter:
        print_key = 1
def on_release(key):
    global press_key,print_key
    if key is keyboard.Key.space:
        press_key = 0
    if key is keyboard.Key.enter:
        print_key = 0
def key_listen():
    with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
         listener.join()
###############

#建立tensorflow模型
def Setup_model():
    model = keras.Sequential([
    keras.layers.Conv2D(4,(3,3),           
                        padding="same",
                        activation="relu"),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(8,(5,5),          
                        padding="same",
                        activation="relu"),
    keras.layers.MaxPool2D((2,2)),#16*16
    keras.layers.Conv2D(16,(5,5),          
                        padding="same",
                        activation="relu"),
    keras.layers.MaxPool2D((3,3)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(130),
    keras.layers.Dense(130),
    keras.layers.Dense(80, activation='tanh'), #这里尝试使用tanh激活函数
    keras.layers.Dense(1)             
])

    model.compile(optimizer='nadam',
                  loss="mse",
                  metrics=['accuracy'])
    
    return model


#主体部分
def run_game(mode=0):
    global press_key,esc,print_key
    truepath = os.getcwd()

    img_path = truepath + "\\trainData\\" + 'images.npy'
    result_path = truepath + "\\trainData\\" + 'result.npy'
    
    try:
        open(img_path)
        open(result_path)
    except FileNotFoundError:
        folder = os.path.exists(truepath + "\\trainData\\")
        if not folder:
            os.makedirs(truepath + "\\trainData\\")
        img = np.zeros(shape=[1,96,96,1])
        result = np.array([0])
        np.save(img_path,img)
        np.save(result_path,result)
        
    if mode == 0:
        
        #使用鼠标获取窗口
        print("[INFO] 请用鼠标点击第一个坐标:")
        x,y = click_pos()
        print("[INFO] [1]坐标:",x,y)
        print("[INFO] 请用鼠标点击第二个坐标:")
        x2,y2 = click_pos()
        print("[INFO] [2]坐标:",x2,y2)

        img = np.load(truepath + "\\trainData\\" + 'images.npy')
        result = np.load(truepath + "\\trainData\\" + 'result.npy')

        t1 = threading.Thread(target=key_listen)
        t1.setDaemon(True)
        t1.start()
        print("[INFO] 按下Space以开始捕获截图,Esc以退出捕获")
        while True:
                if press_key == 0.5:
                    print("[INFO] 开始捕获截图")
                    press_key = 0
                    break

        images = 0
        
        while True:
            img_ready = ImageGrab.grab(bbox=(x, y, x2, y2),include_layered_windows=False, all_screens=True)
            image = img_ready.resize((96,96))
            image = image.convert("L")
            image = np.array(image)
            image = np.expand_dims(image,-1)
            image = np.expand_dims(image,0)
            image = image/255
            img = np.concatenate((img,image),axis=0)
            result = np.append(result,press_key)
            print("[INFO] 已捕获到截图")
                    
            if esc == 1:
                    esc = 0
                    break                                                    
            clock.tick(10)
        
        image = np.delete(image,0,axis=0)
        print("[INFO] 已退出模式[0]")
        np.save(img_path, img)
        np.save(result_path, result)
        esc = 0
        press_key = 1
        run = 1
        
    elif mode == 1:
        train_path = truepath + "\\trainData\\" + 'train.h5'
        if not os.path.exists("model.h5"):
            print("[INFO] 模型正在生成！")
            model = Setup_model()
            print("[INFO] 模型已生成！")
        else:
            model = keras.models.load_model('model.h5')
            print("[INFO] 模型已被读取！")
        
        while True:
            text = input("是否使用数据训练模型？(Y/N):")
            if text == "Y":
                back = 0
                break
            else:
                back = 1
                break
            
        if back != 1:

            img = np.load(truepath + "\\trainData\\" + 'images.npy')
            result = np.load(truepath + "\\trainData\\" + 'result.npy')

            epochs = int(input("训练轮数:"))
            batch_size = int(input("BatchSize:"))
            
            hist = model.fit(img, result, epochs=epochs,batch_size=batch_size)

            print("[INFO] 模型训练完成！")
            run = 1
            model.save('model.h5')
            
            text = input("[INFO] 是否查看训练日志？(Y/N):")
            if text == "Y":
                #使用plt绘制训练日志
                train_acc = hist.history['accuracy']
                train_loss = hist.history['loss']
                
                epochs = range(1, len(train_acc)+1)
                plt.plot(epochs, train_acc, 'bo', label = 'Training acc')
                plt.plot(epochs, train_loss, 'r', label = 'Validation acc')
                plt.title('Training and validation accuracy')
                plt.legend()

                plt.show()
            model.save('model.h5')
            
    elif mode == 2:

        control = keyboard.Controller()
        
        if not os.path.exists("model.h5"):
            print("[ERROR] 模型不存在！请先使用Mode1生成模型！")
        else:
            model = keras.models.load_model('model.h5')
            print("[INFO] 已读取模型")
            
            print("[INFO] 请用鼠标点击第一个坐标:")
            x,y = click_pos()
            print("[INFO] [1]坐标:",x,y)
            print("[INFO] 请用鼠标点击第二个坐标:")
            x2,y2 = click_pos()
            print("[INFO] [2]坐标:",x2,y2)
            
            t1 = threading.Thread(target=key_listen)
            t1.setDaemon(True)
            t1.start()
            print("[INFO] 按下Space以开始捕获截图,ESC以退出")
            while True:
                    if press_key != 2:
                        print("[INFO] 开始捕获截图")
                        press_key = 1
                        break
            while True:

                if esc == 1:
                    esc = -1
                    print("[INFO] 已退出模式[2]")
                    break
                time1 = time.time()
                img_ready = ImageGrab.grab(bbox=(x, y, x2, y2),include_layered_windows=False, all_screens=True)
                image = img_ready.resize((96,96))
                image = image.convert("L")
                image = np.array(image)
                image = np.expand_dims(image,0)
                image = np.expand_dims(image,-1)
                image = image/255
                
                predict = model.predict(image)
                #time1 = time.time()
                if predict[0][0] >= 0.15: #此处只要模型预测出的结果大于0.15就跳跃
                    pag.keyDown("space")
                    print("[INFO] SPACE")
                else:
                    pag.keyUp("space")
                    print("[INFO] NO SPACE")
                    
                print("[INFO] 已捕获到截图")
                print("概率:",predict[0][0])
                time2 = time.time()
                print("FPS:",1/(time2-time1))
        run = 1
            
            
##########################
mode = 1
run = 1
while True:
    print("===================================")
    print("|模式0: 录制训练模型所需的图像数据|")
    print("|模式1: 生成模型并利用图像数据训练|")
    print("|模式2: 开始运行该模型(启动AI)    |")
    print("===================================")
    mode = int(input("请输入启动模式(0/1/2):"))
    run = 1
    if run == 1:
        run = 0
        run_game(mode=mode)
