from ultralytics import YOLO
import cv2
import winsound #für Audioausgabe
import os #für Audioausgabe via MP3-Dateien
import time #für Sleep während Audioausgabe
import numpy as np
import math
import pandas as pd

path='/music' #Pfad für MP3-Dateien

#load camera from csv
df = pd.read_csv("calibration_camera.csv")

camera=df["Camera"][0]
cap = cv2.VideoCapture(int(camera))

# Load custom trained YOLOv8 model
model = YOLO('ai-model-swimmer.pt')
#face/model-face.pt
#ai-model-swimmer.pt
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

show_boxes = True

#load point values from csv
df = pd.read_csv("calibration_points.csv")
B1_x=df["x"][0]
B1_y=df["y"][0]
B2_x=df["x"][1]
B2_y=df["y"][1]
M_x=df["x"][2]
M_y=df["y"][2]
S_x=df["x"][3]
S_y=df["y"][3]

#for counting the number of tracks out of csv
for j in df['Point']:
    j
n=int(j)

#create xlist_lines out of csv
xlist_lines=[B1_x]
l=4
while(l<n+4):
    xlist_lines.append(df['x'][l])
    l+=1
#create ylist_lines out of csv
ylist_lines=[B1_y]
l=4
while(l<n+4):
    ylist_lines.append(df['y'][l])
    l+=1

xlist_lines_center = []
ylist_lines_center = []
swimmer_difference_to_track_list = []

out = cv2.VideoWriter('output1.mp4', fourcc, fps,(frame_width,frame_height),True )
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        swimmer_found = False
        results = model(frame, imgsz=640, stream=True, verbose=False)
        
        for result in results:
            
            for box in result.boxes.cpu().numpy():
                
                
                if show_boxes:
                    name=result.names[int(box.cls[0])]
                    r = box.xyxy[0].astype(int)
                    c= box.conf[0] #confidence of detection
                    if c>0.3:
                        x=r[0]
                        y=r[1]
                        
                        print(c, x, y)

                        """
                        #--Bahnerkennung--
                        z=0
                        j=0
                        while (z<len(xlist_lines)-1):
                            xlist_lines_center.append(xlist_lines[z]+((xlist_lines[z+1]-xlist_lines[z])/2))
                            ylist_lines_center.append(ylist_lines[z]+((ylist_lines[z+1]-ylist_lines[z])/2))
                            z+=1

                        while (j<len(xlist_lines_center)):
                            swimmer_difference_to_track_list.append( math.sqrt( math.pow(x-xlist_lines_center[j],2) + math.pow(y-ylist_lines_center[j],2) ) )    
                            j+=1
                        
                        swimmer_track_nr=swimmer_difference_to_track_list.index(min(swimmer_difference_to_track_list))+1
                        
                        print(swimmer_difference_to_track_list)
                        print("Schwimmer Bahn-Nr.", swimmer_track_nr)
                        
                        #--Geschwindigkeitserkennung
                        xlist_last_position = [2000]*n
                        ylist_last_position = [2000]*n
                        list_last_time = [time.time()]*n
                        list_swimmer_act_speed=[0]*n
                        list_need_time = [4]*n # 4 Sekunden Abstand zur Initialisierung aller Bahnen
                        
                        
                        '''
                        #Test erforderlich
                        list_swimmer_act_speed[swimmer_track_nr-1]=(math.sqrt( math.pow(x-xlist_last_position[swimmer_track_nr-1],2) + math.pow(y-xlist_last_position[swimmer_track_nr-1],2) ) ) / (time.time() - list_last_time[swimmer_track_nr-1]) #-1, da ja Liste bei Index 0 beginnt
                        print("Aktuelle Geschwindigkeitsliste in Pixel/s", list_swimmer_act_speed)
                        list_need_time[swimmer_track_nr-1]=swimmer_difference_to_track_list[swimmer_track_nr-1] / list_swimmer_act_speed[swimmer_track_nr-1]
                        print("Voraussichtliche Ankunftszeitliste in Sekunden ", list_need_time)
                        '''
                        
                        #--Rücksetzung Bahnerkennung--
                        z=0
                        j=0
                        xlist_lines_center = []
                        ylist_lines_center = []
                        swimmer_difference_to_track_list = []
                        """
                        #for pool edge line r(x)>k_r*x+d_r
                        k_r=(B2_y-B1_y)/(B2_x-B1_x)
                        d_r=B1_y-(k_r*B1_x)
                        
                        #for pool line n(x)>k_n*x+d_n
                        k_n=(M_y-B2_y)/(M_x-B2_x)
                        d_n=B2_y-(k_n*B2_x)
                        
                        #for pool line with B1 t0(x)<k_n*x+d_t0
                        d_t0=B1_y-(k_n*B1_x)
                        
                        #for signal line  s(x)<k_r*x+d_s
                        d_s=M_y-(k_r*M_x)
                        
                        """
                        #single feedback (easy version)
                        if  y<k_r*x+d_s:
                            winsound.Beep(500, 3000) #erster Wert=Frequenz - Zweiter Dauer in ms
                        #single feedback (version 2)
                        if  y<k_r*x+d_s and y>k_r*x+d_r:
                            winsound.Beep(500, 3000)
                        """
                        #single feedback (version 3)
                        if  y<k_r*x+d_s and y>k_r*x+d_r and y>k_n*x+d_n and y<k_n*x+d_t0:
                            winsound.Beep(500, 3000)
                        

                        
                cls = int(box.cls[0])
                
                if cls == 0:
                    swimmer_found = True
                    

        if swimmer_found:
            out.write(frame)
        
        cv2.imshow('SWIM-ASSIST', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# for openCV - When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()