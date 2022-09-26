import pyautogui
import pyperclip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import time
labels = []



for i in range(1000000):
    print(f'현재 단계 {i}')
    while True:
        if keyboard.is_pressed('space'):
            break
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.5)
    xmin = pyperclip.paste()
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.5)
    xmax = pyperclip.paste()
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.5)
    ymax = pyperclip.paste()
    time.sleep(0.5)
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.5)
    ymin = pyperclip.paste()
    time.sleep(0.5)
    print(f'xmin: {xmin}')
    print(f'ymin: {ymin}')
    print(f'xmax: {xmax}')
    print(f'ymax: {ymax}')

    labels.append([xmin, ymin, xmax, ymax])

    labels_result = pd.DataFrame(labels)
    labels_result.columns = ['xmin','ymin','xmax','ymax']
    labels_result.to_csv('C:\\Users\\userr\\Desktop\\도면10레이블.csv', index = False, encoding = 'utf-8-sig')
    print('단계 끝')
    print()
