# Project - PyQt and DCP
![result](/assets/result.bmp)

## Introduction
[Dark Channel Prior (DCP)](http://kaiminghe.com/cvpr09/index.html): 2009年 Kaiming He 於 CVPR 上發表的 paper，也榮獲 2009 CVPR Best Paper Award。

其中 DCP 廣為人知的缺點為對於天空的透射圖 (Transmission Map) 會過度估計，我也針對此缺點做了天空保留 (Sky Preservation)。

## Requirement
* Ubuntu 16.04
* Python 3.6
* Python Packages  
\- PyQt5  
\- functools  
\- numpy  
\- cv2  
\- ctypes  
* gcc 5.4.0  
\- pthread  
\- Intel SSE2  

## Demo
```
$ python3 main.py
```

## Parameters
如附圖，此程式提供 7 個可調控參數：
* Min Radius  
Range: 7 ~ 107  
控制 Minimum Filter 的半徑大小  
* Box Radius  
Range: 7 ~ 107  
控制 Box Filter 的半徑大小  
* Omega  
Range: 0.01 ~ 1.00  
調控整體除霧程度  
* T0  
Range: 0.01 ~ 1.00  
Transmission map 的下限值  

### Parameters for Sky Preservation
藉由以下兩個參數達到天空保留的效果
* Variance  
Range: 0.01 ~ 2.00  
調控天空區域與非天空區域的閥值  
* T1  
Range: 0.01 ~ 1.00  
調控天空區域的 Transmission 下限值  

### Airlight Estimation
* AL Offset  
Range: 1 ~ 256  
設定大氣光採樣的閥值  
Threshold = 256 - AL_Offset  

## Project Description
做個 Project 的初衷是讓調整參數時可以更即時看到效果，不用每次改一個參數就要重新 compile 一次。經過這個 Project 也學習到 PyQt 做 GUI 和藉由 ctypes 使用 FFI。

此程式的構成可分為三部分：
* DCP: 使用 C 語言撰寫，其中使用 pthread、SSE 和一些小技巧來加速演算法
* GUI: 使用 python + PyQt 撰寫，參考至 [zetcode - PyQt5 tutorial](http://zetcode.com/gui/pyqt5/)
* FFI: 使用 ctypes 連接 Python 與 C

前前後後花了差不多兩個禮拜，尤其是 GUI 是從零開始學，layout 的部分是手動的，之後應該會嘗試看看用 Qt Designer 來 layout。另外，ctypes 的使用上，因為不熟也碰到很多坑，這個真的是最頭痛的部分，不過最終也完成了這個 project。

> 因為 C code 的部分需要保密，所以我只提供 `.so` 檔  
> Sorry for this. :(
