import pyautogui
import cv2
import numpy as np
from yolov3 import yolov3
import math
import random


from pynput import keyboard

import time

    
    
def closest_detection(targets, pivot, class_id, high_range=1):    #targets is list [x1,y1,x2,y2,... class_id]     pivot is list [x,y]

    #chooses a random detection from the top 'high_range" objects
    target_locs = []
    h_range = high_range

    
    #finds detected objects with class_id and calculates distance from pivot
    for obj in targets:
        if obj[6] == class_id:
            center_x = obj[0] + ((obj[2] - obj[0])//2)
            center_y = obj[1] + ((obj[3] - obj[1])//2)
        
            #distance from pivot
            dist = math.sqrt(((pivot[0] - center_x)**2) + ((pivot[1] - center_y)**2))
            
            new_target = [center_x, center_y, class_id, dist]

            target_locs.append(new_target)    
    
    #sorts objects on distance
    target_locs.sort(key = lambda locs: locs[3])
    
    #checks if the top n values exists 
    if len(target_locs) < high_range or not target_locs:
        h_range = len(target_locs)

    #check is there are any targets exist with class_id
    if len(target_locs) > 0:
        return target_locs[random.randint(0, h_range - 1)][0:3]
    else:
        return 0

def health_from_image(heart_icon):
    rolling_sum = [0,0,0]
    center_x = heart_icon.width//2
    for h in range(heart_icon.height):
        current_pixel = heart_icon.getpixel((center_x, h))
        rolling_sum[0] = rolling_sum[0] + current_pixel[0]

    return rolling_sum[0]



COLOR_THRESHOLD = []
with open("config/color_mask_config.cfg") as color_mask_config:
    color_thres_list = color_mask_config.read()
    color_thres_list = color_thres_list.split("\n")

for object_id in color_thres_list:
    for color in object_id:
        COLOR_THRESHOLD.append(object_id.split(","))


##################
def color_mask_detect(frame):
    detections = []

    for object_type in COLOR_THRESHOLD:
        unrolled_frame = np.array(frame)
        unrolled_frame = unrolled_frame.ravel()

        color_filter = np.full((frame.width, frame.height,3), object_type).ravel()
        
        sieved_pixels = np.abs(np.subtract(color_filter, unrolled_frame))
        

    return []
    #For pixel in frame:
        

cfg = "config/yolov3-tiny-rs.cfg"
weights = "weights/yolov3-tiny-rs.weights"
classes = "config/rs.names"

#image_path = "images/3.png"



print("\nInitializing yolov3 parameters...\n")

yolo = yolov3(cfg, classes, weights, nms_thres=0.8)


iter = 0
no_detect_flag = False


#timeout parameters
start_time = time.perf_counter()
timeout_thres = 18000



#ADHOC TO EMPTY INVENTORY
friends_button = None
while friends_button is None:
    friends_button = pyautogui.locateOnScreen('friends_button.png')
    if not friends_button:
        print("Cannot locate friends button       ", end="\r")
friends_button = (friends_button[0] + (friends_button[2]),friends_button[1] + (friends_button[3]//2) - 235)

'''
rs_tab = None
while rs_tab is None:
    rs_tab = pyautogui.locateOnScreen('logo.png')
    if not rs_tab:
        print("Cannot locate runescape tab       ", end="\r")
rs_tab = (rs_tab[0] + (rs_tab[2]), rs_tab[1] + (rs_tab[3]//2))
'''


targetables = None
with open("config/rs.names") as file:
    targetables = file.read()
    targetables = targetables.split("\n")
    targetables.pop()   #names file has blank end

    print("Target Options")
    for targ in enumerate(targetables):
        print(str(targ[0]) + ": " + targ[1])



print("Enter target id: ", end = "")
target_id = int(input())


print("-------" + targetables[target_id] + "hunting autonomy active-------")

while True:
    try:
        screen = pyautogui.screenshot()
        
        screen.resize(size = (800,600))
        output = yolo(screen)
        #output = color_mask_detect(screen)
        
        

        #fix to occur only once
        character_loc = [screen.width//2, screen.height//2]


        if iter % 8 == 0 and target_id == 0:
                print("Clearing inventory...                                         ", end="\r")

                
                pyautogui.keyDown('shift')

                
                click_loc_init = (friends_button[0], friends_button[1])
                click_loc = list(click_loc_init)

                for ind in range(24):
                    pyautogui.moveTo(click_loc[0], click_loc[1], 0.1 + (random.random() * 0.1), pyautogui.easeInCubic)
                    time.sleep(0.05+random.random()*0.15)
                    #check if mouse is moving

                    if pyautogui.position() != tuple(click_loc):
                        pyautogui.keyUp('shift')
                        break
                    
                    pyautogui.click()
                    time.sleep(0.2)

                    click_loc[0] += 42

                    if (ind + 1) % 4 == 0:
                        click_loc[0] = click_loc_init[0]
                        click_loc[1] += 39

                #close bag ui
                pyautogui.moveTo(1096,838,0.2,pyautogui.easeInCubic)
                pyautogui.click()

                
                for obj in output:
                    if obj[6] == 1:
                        center_x = obj[0] + ((obj[2] - obj[0])//2)
                        center_y = obj[1] + ((obj[3] - obj[1])//2)

                        
                        pyautogui.moveTo(center_x, center_y, 0.5, pyautogui.easeInCubic)
                        time.sleep(0.1)
                        pyautogui.click()
                        time.sleep(0.1)
                        pyautogui.moveTo(pyautogui.position()[0] + int(random.random() * (20)), pyautogui.position()[1] + int(random.random() * (20)))
                        time.sleep(0.4)
                

                pyautogui.keyUp('shift')

        
        if output:
            
            target = closest_detection(output, character_loc, target_id, 1) #second to last parameter is targeted class_id
            
            if target:

                pyautogui.moveTo(target[0], target[1], 0.1, pyautogui.easeInCubic)
                time.sleep(0.1)
                pyautogui.click()
                time.sleep(18)

                n_clicks = random.randint(1,3)
                for click_n in range(n_clicks):
                    pyautogui.moveTo(character_loc[0] + random.randint(-20,20), character_loc[1] + random.randint(-20,20), 0.5/n_clicks, pyautogui.easeInCubic)

                print("Targeting " + targetables[target_id] +"                                   ", end="\r")

                
            #print("Round " + str(it_round) + " number of objects detected: " + str(len(output)), end="\r")
            else:
                print("No targets detected                                   ", end="\r")

            #rotates camera to bring more targets into vision
            pyautogui.keyDown('right')
            time.sleep(random.random())
            pyautogui.keyUp('right')

            #estimated time to complete action
            time.sleep(2)

            #finds detected logs and click on them to drop
            
            
            
            iter += 1
            no_detect_flag = False

        else:
            #if no detections click on precomputed runescape icon location to maximize
    
            if not no_detect_flag:
                #pyautogui.click(rs_tab[0], rs_tab[1])
                #   
                #rotates camera to bring more targets into vision
                pyautogui.keyDown('right')
                time.sleep(random.random())
                pyautogui.keyUp('right')


            no_detect_flag = True
            


        #timeout timer
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > timeout_thres:
            print("Timeout exit...                                         ")
            quit()
        else:
            print("Iteration no." + str(iter) + ": " + str(int(timeout_thres - elapsed_time)) + " seconds until timeout", end = "\r")

        #temporary healing fix
        if elapsed_time % 300 <= 5:
            print("Healing...                                              ", end = "\r")
            time.sleep(100)


    except KeyboardInterrupt:
        print("Bot paused. Enter y to continue", end = "\r")
        f = input()
        if f != 'y':
            break

print("Exiting...                                                             ")