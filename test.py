import cv2
import sys
import numpy as np
import subprocess
 
# 背景差分版
def capture(cap):
    ret,frame = cap.read()
    return cv2.resize(frame, (320, 240))
 
def mark(mask, frame):
        ref = 0
        # 動いているエリアの面積を計算してちょうどいい検出結果を抽出する
        thresh = cv2.threshold(mask, 3, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        areaf = 0
        if (contours):
            target = contours[0]
        for cnt in contours:
             #輪郭の面積を求めてくれるcontourArea
            area = cv2.contourArea(cnt)
            if max_area < area and area < 10000 and area > 800: 
                max_area = area;
                target = cnt
            # 動いているエリアのうちそこそこの大きさのものがあればそれを矩形で表示する
            if (max_area <= 800):
                areaf = frame
            else:
                # 諸般の事情で矩形検出とした。
                x,y,w,h = cv2.boundingRect(target)
                areaf = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                ref = 1
 
        return ref, areaf
 
def main():
 
    i = 0      # カウント変数
    # カメラのキャプチャ
    cap = cv2.VideoCapture('video/video08.mp4')
    if cap.isOpened() is False:
        print("can not open camera")
        sys.exit()
 
    cap.set(cv2.CAP_PROP_FPS, 10)
    
    # 最初のフレームを背景画像に設定
    bg = capture(cap)
    fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
 
    # グレースケール変換
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) 
    ref = 0
    skip_count = 0
    while(cap.isOpened()):
        # フレームの取得
        frame = capture(cap)
 
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)
        cv2.imshow("mask", mask)
        if( skip_count == 0 ):
            ref, areaframe = mark(mask, frame)
            # フレームとマスク画像を表示
            cv2.imshow("areaframe", areaframe)
        skip_count = (skip_count + 1) % 30
 
        i += 1    # カウントを1増やす
        if (ref):
           i = 0
           ref = 0
 
        # 背景画像の更新（一定間隔）
        if(i > 10):
            bg = capture(cap)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) 
            i = 0 # カウント変数の初期化
 
        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()