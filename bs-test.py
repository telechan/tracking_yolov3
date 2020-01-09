import numpy as np
import cv2
import os

def run_KNN(video_path, out_path=""):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorKNN()
    # if not cap.isOpend():
    #     raise IOError("Couldn't open!")
    
    vid_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(vid_size)
    
    isOutput = True if out_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(out_path), type(vid_fourcc), type(vid_fps), type(vid_size))
        out = cv2.VideoWriter(out_path, vid_fourcc, vid_fps, vid_size)
    
    while True:
        ret, frame = cap.read()
        if type(frame) == type(None): break
        mask = fgbg.apply(frame)
        # bg = fgbg.getBackgroundImage()
        # cv2.imshow('result mask', mask)
        # cv2.imshow('result bg', bg)

        thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('Threshold Frame', thresh)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if hierarchy[0][i][3] == -1 and area > 50:
                    result_img = cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 2)
        cv2.imshow('result image', result_img)

        if isOutput:
            out.write(result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
        
def select_in_out():
    finished = True
    while finished:
        input_path = input('Input video path:')
        output_path = input('Output video path:')
        try:
            run_KNN(input_path, output_path)
            finished = False
        except:
            print('Error! Try again')
            continue
    
if __name__ == "__main__":
    select_in_out()