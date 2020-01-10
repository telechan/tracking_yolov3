import numpy as np
import cv2
import os

def run_KNN(video_path, out_path=""):
    fgbg = cv2.createBackgroundSubtractorKNN()
    fgbg2 = cv2.bgsegm.createBackgroundSubtractorGSOC()

    if video_path.isdigit():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps       = cap.get(cv2.CAP_PROP_FPS)
    video_size      = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if out_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(out_path, video_FourCC, video_fps, (video_size[0] * 2, video_size[1] * 2), 0)
    print(video_size)
    
    while True:
        ret, frame = cap.read()
        if type(frame) == type(None): break

        cv2.namedWindow('bs window', cv2.WINDOW_NORMAL)
        cv2.namedWindow('area window', cv2.WINDOW_NORMAL)

        mask = fgbg.apply(frame)
        # cv2.imshow('result mask', mask)

        mask2 = fgbg2.apply(frame)
        # cv2.imshow('result mask', mask2)

        thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('Threshold Frame', thresh)

        thresh2 = cv2.threshold(mask, 3, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('Threshold Frame', thresh2)

        img_KNN = cv2.hconcat([mask, thresh])
        img_GSOC = cv2.hconcat([thresh2, mask2])

        contours1, hierarchy1 = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours1):
            area = cv2.contourArea(cnt)
            if hierarchy1[0][i][3] == -1 and area > 1000:
                    result_img1 = cv2.drawContours(frame.copy(), [cnt], 0, (255, 0, 0), 2)
        # cv2.imshow('thresh1 image', result_img1)

        contours2, hierarchy2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours2):
            area = cv2.contourArea(cnt)
            if hierarchy2[0][i][3] == -1 and area > 1000:
                    result_img2 = cv2.drawContours(frame.copy(), [cnt], 0, (255, 0, 0), 2)
        # cv2.imshow('thresh2 image', result_img2)

        contours3, hierarchy3 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours3) == 0:
            result_img3 = frame
        for i, cnt in enumerate(contours3):
            area = cv2.contourArea(cnt)
            if hierarchy3[0][i][3] == -1 and area > 1000:
                    result_img3 = cv2.drawContours(frame.copy(), [cnt], 0, (255, 0, 0), 2)
        # cv2.imshow('GSOC image', result_img3)

        image1 = cv2.hconcat([result_img1, result_img2])
        image2 = cv2.hconcat([result_img3, frame])

        result_image = cv2.vconcat([img_KNN, img_GSOC])
        result_image2 = cv2.vconcat([image1, image2])
        cv2.imshow('bs window', result_image)
        cv2.imshow('area window', result_image2)

        if isOutput:
            out.write(result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
        
def select_in_out():
    # input_path = input('Input video path:')
    # output_path = input('Output video path:')
    run_KNN('video/video08.mp4', '')
    # finished = True
    # while finished:
    #     input_path = input('Input video path:')
    #     output_path = input('Output video path:')
    #     try:
    #         run_KNN(input_path, output_path)
    #         finished = False
    #     except:
    #         print('Error! Try again')
    #         continue
    
if __name__ == "__main__":
    select_in_out()