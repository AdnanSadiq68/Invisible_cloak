import cv2 # for image processing 
import numpy as np # This is a mathematical library for image handling

cap = cv2.VideoCapture(0)
background = cv2.imread('./image.jpg')

while cap.isOpened():
    # capture the live frame
    ret, current_frame = cap.read()

    # HSV color space represents colors using 3 values
    # its angle range is from 0 to 360 degrees
    # 0 degree corressponds to RED colour, 120 degree corresponds to GREEN colour, 240 degree corresponds to BLUE colour
    # Saturation represents the intensity of colours
    if ret: 
        # converting image from rgb to hsv color space
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # red color is represented by 0 to 10 and 170 to 180 values
        # the actual value range between 0 to 360, But in opencv to fit into 8 bit value the range is from 0 to 180 degrees

        # Range for lower red 
        l_red = np.array([0,120,70])
        u_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv_frame, l_red, u_red)

        #range for upper red 
        l_red = np.array([170,120,70])
        u_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv_frame, l_red, u_red) 

        # generating the final red mask
        red_mask = mask1 + mask2

        # edges are not precise and output are having some noraml bleaches so to overcome this we have in opencv somrthing called as
        # morphology
        # This line of code will remove all kinds of small false detections which avoids the normal bleaches in the output
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations= 10)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations= 1)

        # substituting the red portion with background image
        part1 = cv2.bitwise_and(background, background, mask = red_mask)

        #detecting the things that are not red
        red_free = cv2.bitwise_not(red_mask)

        # if cloak is not present show the current image
        part2 = cv2.bitwise_and(current_frame, current_frame, mask = red_free)


        #final output by combining the part1 and part2
        cv2.imshow("Cloak",part1 + part2)
        if cv2.waitKey(5) == ord('q'):
           break
cap.release()
cv2.destroyAllWindows()