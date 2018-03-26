import cv2
import numpy as np

import argparse

def video_iterator(vid):
    v = cv2.VideoCapture(vid)
    _, frame = v.read()
    while frame is not None:
        yield frame.astype(float)[...,1]
        _, frame = v.read()

def get_background(vid_stream, n, accum_func=np.mean):
    bg = np.array([next(vid_stream) for i in range(n)])
    return accum_func(bg, axis=0)

def track(vid_stream, bg, threshold, target_area, opening_radius, 
          debug=False):
    kernel = np.zeros((opening_radius, opening_radius))
    c = opening_radius / 2.
    for i in range(opening_radius):
        for j in range(opening_radius):
            if (i - c) ** 2 + (j - c) ** 2 <= opening_radius ** 2:
                kernel[i, j] = 1

    for frame in vid_stream:
        diff = frame - bg
        th = (diff < threshold).astype("uint8") * 255
        seg = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        seg = seg.astype("uint8")
        if debug: 
            cv2.imshow("bg", bg.astype('uint8'))
            cv2.imshow("orig", frame.astype('uint8'))
            cv2.imshow("threshold", th)
            cv2.imshow("seg", seg)
        _, contours, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            yield None
            continue

        #select contour
        contour_area = np.array([cv2.contourArea(c) for c in contours])
        contour_area = np.abs(contour_area - target_area)
        select_contour = np.argmin(contour_area)

        moments = cv2.moments(contours[select_contour])
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])

        disp_frame = frame.copy().astype("uint8")
        disp_frame = np.dstack([disp_frame, disp_frame, disp_frame])
        cv2.drawContours(disp_frame, contours, select_contour, 
                                                (0, 255, 0), 1)
        cv2.circle(disp_frame, (cx, cy), 3, (0, 0, 255), thickness=-1)
        yield (cx, cy), disp_frame




def get_args():
    parser = argparse.ArgumentParser(description="Spatial stimulator")
    parser.add_argument("input", help="input video source")
    parser.add_argument("-b", "--background", 
            help="number of frames to image for background extraction.",
            type=int, default=1)
    parser.add_argument("--median", 
            help="use median instead of mean for background",
            action="store_true")
    parser.add_argument("-a", "--target-area",
            help="total area of mouse in pixel square",
            type=float, default=550)
    parser.add_argument("-t", "--threshold",
            help="threshold for segmentation", type=int, default=-50)
    parser.add_argument("-r", "--binary-opening-radius",
            help="binary opening radius for segmentation",
            type=int, default=4)
    parser.add_argument("-d", "--debug", help="show debug information",
            action="store_true")
    parser.add_argument("-q", "--quiet", help="show nothing",
            action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    vid_stream = video_iterator(args.input)
    accum_func = np.mean
    if args.median:
        accum_func = np.median
    bg = get_background(vid_stream, args.background, accum_func)
    position_stream = track(vid_stream, bg, args.threshold, 
                            args.target_area,
                            args.binary_opening_radius,
                            debug=args.debug)

    for res in position_stream:
        if not res:
            print(".")
            continue
        coord, img = res
        if not args.quiet:
            cv2.imshow("tracking", img)
            cv2.waitKey(1)
        print(coord)

    


