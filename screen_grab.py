import cv2
import time
import mss # for screen grab
import numpy as np
from lane_detection import LaneDetection

class GrabScreen(object):

    def __init__(self, 
                    monitor_dims={'top': 40, 'left': 0, 'width': 800, 'height': 600}, 
                    show_fps=True, 
                    mouse_callback=False):

        # Coordinates of monitor screen to grab
        self.monitor = monitor_dims
        # Display FPS (True or False)
        self.show_fps = show_fps
        # if TRUE, DOUBLE-CLICK on grabed screen to get pixel coordinates
        self.mouse_callback = mouse_callback

    # get coordinates of double click on image
    mouse_click_coords = []
    def mouse_coords(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("MOUSE POS---------->({}, {})".format(x, y))
            mouse_click_coords.append([x, y])
            print(mouse_click_coords)

    def grab(self):
        global window_search 
        window_search = True

        # if TRUE, set mouse callback function        
        if self.mouse_callback:
            cv2.namedWindow("screen_grab")
            cv2.setMouseCallback("screen_grab", mouse_coords)

        # create lane detection class instance
        find_lane = LaneDetection()

        with mss.mss() as sct:
            while 'Screen capturing':
                last_time = time.time()
                # Get raw pixels from the screen, save it to a Numpy array
                img = np.array(sct.grab(self.monitor))

                try:
                    processed_img = find_lane.get_image(img)
                    cv2.imshow('screen_grab', processed_img)
                except:
                    print("Finding Lanes...")
                    pass

                if self.show_fps:
                    print('FPS: {0:.0f}'.format(1/(time.time()-last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", default=True, help='Set True to display FPS')
    GrabScreen(show_fps=parser.fps).grab()