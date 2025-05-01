import cv2
import mediapipe as mp
import time 
import os
import hand_tracking_module as htm


w_cam , h_cam = 640, 480    ## setting webcam's widtha and height

cap = cv2.VideoCapture(0)  # Try different indices if needed (0, 1, 2, ......)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, w_cam)
cap.set(4, h_cam)

time.sleep(2)  # Give the camera time to initialize

p_time = 0
c_time = 0

## reading images from the folder
folder_path = "hand_finger"
my_finger_list = os.listdir(folder_path)
print(my_finger_list)

## loading images into the list
overlay_list = []
for img_path in my_finger_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    # print(f'{folder_path}/{img_path}')
    overlay_list.append(image)
    
# print(len(overlay_list))
    
# Resize all images in overlay_list **before the loop**
target_size = (200, 200)
overlay_list_resized = [cv2.resize(img, target_size) for img in overlay_list]




detector = htm.hand_detector(detection_con=0.75)

tip_ids = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    
    if not success or img is None:
        print("Error: Failed to capture image")  
        break  # Exit if no frame is captured
    
    
    img = detector.find_hands(img)
    lm_list, b_box = detector.find_position(img, draw = False)
    
    
    total_fingers = 0
    # print(lm_list)
    if len(lm_list) != 0:
        fingers = []
        
        ## ✅ Thumb Check: Compare x-coordinates for **left and right hands**
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0]-1][1]:  # Right hand
            fingers.append(1)
        else:
            fingers.append(0)
        
        ## this will work for the four fingers except thumb finger
        for id in range(1, 5):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        
        total_fingers  = fingers.count(1)
        print(total_fingers)
        
        # If all fingers are closed, show "6.jpg" but keep the count as 0
        if total_fingers == 0:
            img[0:200, 0:200] = overlay_list_resized[-1]  # Show last image (6.jpg)
            total_fingers_display = 0  # Keep displayed count as 0
        else:
            total_fingers_display = total_fingers

    # Ensure the index is valid before accessing the overlay list
    if 0 < total_fingers <= len(overlay_list_resized):
        img[0:200, 0:200] = overlay_list_resized[total_fingers - 1]  # Display only valid finger counts
        
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers_display), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
    
    
    c_time  = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    
    cv2.putText(img,f'FPS: {str(int(fps))}', (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit
    