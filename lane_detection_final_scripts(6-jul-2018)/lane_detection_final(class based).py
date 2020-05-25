import time
import cv2
import mss
import numpy as np

class LaneDetection(object):
    """
    Detect road lanes in GTA5 video game
    """
    def compute_binary_image(self, color_image):
        """
        Extract yellow and white lanes from BGR image and
        convert to binary image.
        """
        # get yellow lanes 
        lower_yellow = np.array([0,50, 100])
        upper_yellow = np.array([180, 255, 255])
        mask_yellow = cv2.inRange(color_image, lower_yellow, upper_yellow)
        output_yellow = cv2.bitwise_and(color_image, color_image, mask = mask_yellow)
        gray_yellow = cv2.cvtColor(output_yellow, cv2.COLOR_BGR2GRAY)
        thres = 180
        gray_yellow[(gray_yellow >= thres)] = 255
        gray_yellow[(gray_yellow < thres)] = 0

        # get white lanes
        lower_white = np.array([150, 150, 150])
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(color_image, lower_white, upper_white)
        output_white = cv2.bitwise_and(color_image, color_image, mask = mask_white)
        gray_white = cv2.cvtColor(output_white, cv2.COLOR_BGR2GRAY)
        thres = 150
        gray_white[(gray_white >= thres)] = 255
        gray_white[(gray_white < thres)] = 0

        # combine yellow and white lanes 
        yw_binary = cv2.add(gray_yellow, gray_white)

        return yw_binary

    def region_of_interest(self, img, vertices):
        """
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        mask = np.zeros_like(img)
        # if len(img.shape) > 2:
        #     channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        #     ignore_mask_color = (255,) * channel_count
        # else:
        #     ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, 255)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def compute_perspective_transform(self, binary_image):
            """
            Applies prespective tranform from source image coordinates
            to destination image coordinates
            """
            transform_src = np.float32([[300, 309], [500, 315], [120, 381], [685, 392]])
            transform_dst = np.float32([ [0,0], [800, 0], [0,600], [800,600]])
            perspective_transform = cv2.getPerspectiveTransform(transform_src, transform_dst)
            inverse_perspective_transform = cv2.getPerspectiveTransform(transform_dst, transform_src)
            warped_image = cv2.warpPerspective(binary_image, perspective_transform, 
                                                            (binary_image.shape[1], binary_image.shape[0]), 
                                                            flags=cv2.INTER_NEAREST)

            return warped_image, inverse_perspective_transform


    def track_lanes_initialize(self, binary_warped):
        """
        The preliminary search works like this:
        Create a search window on the bottom of the image whose height is 1/9 of the image's height.
        Split the window into left and right halves.
        Locate the pixel column with the highest value via histogram.
        Draw a box around that area using a margin variable.
        Identify all of the non-zero pixels in that box. If there are enough, center the box on their mean position for the next window.
        Fit a quadradtic equation to all of the non-zero pixels identified in each half of the image (left lane and right lane)
        """
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # we need max for each half of the histogram. the example above shows how
        # things could be complicated if didn't split the image in half 
        # before taking the top 2 maxes
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        # this will throw an error in the height if it doesn't evenly divide the img height
        nwindows = 5
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = int(binary_warped.shape[0] - (window + 1) * window_height)
            win_y_high = int(binary_warped.shape[0] - window * window_height)
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) 
                                  & (nonzeroy < win_y_high) 
                                  & (nonzerox >= win_xleft_low) 
                                  & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) 
                                  & (nonzeroy < win_y_high) 
                                  & (nonzerox >= win_xright_low) 
                                  & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

                
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0] )
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin))
                             & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
                             & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def track_lanes_update(self, binary_warped, left_fit, right_fit):
        """
        Tracking lanes across search windows.
        Once we have the polynomial of the line that best fits the lane,
        we can optimize our search by only looking in the neighborhood
        of that polynomial from frame to frame.
        """
        global window_search
        
        # repeat window search to maintain stability
        window_search=True

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                             & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                             & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


        return left_fit,right_fit,leftx,lefty,rightx,righty

    # A function to get quadratic polynomial output
    def get_val(self, y,poly_coeff):
        return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

    def lane_fill_poly(self, binary_warped,undist, inverse_perspective_transform, left_fit,right_fit):
        """
        Fill area between lanes 
        """
        # Generate x and y values
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.get_val(ploty, left_fit)
        right_fitx = self.get_val(ploty, right_fit)
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast x and y for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane 
        cv2.fillPoly(color_warp, np.int_([pts]), (255,255, 255))

        # Warp using inverse perspective transform
        newwarp = cv2.warpPerspective(color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0])) 
        # overlay
        #newwarp = cv.cvtColor(newwarp, cv.COLOR_BGR2RGB)
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
            
        return result

    def image_processing(self, image):

        # remove alpha channel
        bgr_image = image[:, :, :-1]

        global window_search
        global left_fit_prev
        global right_fit_prev

        # Get binary image from original image
        binary_img = self.compute_binary_image(bgr_image)

        # apply Region of Interest
        imshape = binary_img.shape
        middle_y = .67
        top_x = .30
        top_y = .40
        lower_left = [0, imshape[0]]
        middle_left = [0, imshape[0] * middle_y]
        top_left = [imshape[1] * top_x, imshape[0] * top_y]
        top_right = [imshape[1] - imshape[1] * top_x, imshape[0] * top_y]
        middle_right = [imshape[1], imshape[0] * middle_y]
        lower_right = [imshape[1], imshape[0]]
        vertices = [np.array([lower_left, middle_left, top_left, top_right, middle_right, lower_right], dtype=np.int32)]
        roi_img = self.region_of_interest(binary_img, vertices)

        #Transform image prespective to birds eye view
        warped_img, inverse_perspective_transform = self.compute_perspective_transform(roi_img)

        if window_search:
            #window search
            left_fit, right_fit = self.track_lanes_initialize(warped_img)
            #store values
            left_fit_prev = left_fit
            right_fit_prev = right_fit
            
        else:
            #load values
            left_fit = left_fit_prev
            right_fit = right_fit_prev
            #search in margin of polynomials
            left_fit, right_fit, leftx, lefty, rightx, righty = self.track_lanes_update(warped_img, left_fit, right_fit)
        
        #save values
        left_fit_prev = left_fit
        right_fit_prev = right_fit
        #draw polygon
        processed_frame = self.lane_fill_poly(warped_img, bgr_image, inverse_perspective_transform, left_fit, right_fit)

        return processed_frame

    def get_image(self, image):
        return self.image_processing(image)


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
    GrabScreen().grab()