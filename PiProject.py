import cv2 
import numpy as np
import dlib 
from math import hypot
import time

#which camera to use
cap=cv2.VideoCapture(0) 

#dlib detects the points on shape predictor
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font=cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1,p2):
    return int((p1.x+p2.x)/2),int((p1.y+p2.y)/2)

def get_blinking_ratio(eye_points,facial_landmarks):
    #points for drawing the horizontal and vertical line that make a cross on the eye
    left_point=(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y)
    right_point=(facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y)
    center_top=midpoint(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
    center_bottom=midpoint(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))

    #lines that form a cross on the eye
    #hor_line=cv2.line(frame,left_point,right_point,(0,255,0),2)
    #ver_line=cv2.line(frame,center_top,center_bottom,(0,255,0),2)

    hor_line_length=hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
    ver_line_length=hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))

    #ratio will decise if the eye is blinking or not based on an interval that we adjust depending on how it responds to tests
    ratio=hor_line_length/ver_line_length
    return ratio


def get_gaze_ratio(eye_points,facial_landmarks):
        eye_region=np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x,facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x,facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x,facial_landmarks.part(eye_points[5]).y)],np.int32)

        height,width,_=frame.shape

        #mask that is a black frame with the threshold of the eyes
        mask=np.zeros((height,width),np.uint8)

        #polylines draws around the eyes
        cv2.polylines(frame,[eye_region],True,255,2)
        cv2.fillPoly(mask,[eye_region],255)
        eye=cv2.bitwise_and(gray,gray,mask=mask)

        min_x=np.min(eye_region[:,0])
        max_x=np.max(eye_region[:,0])
        min_y=np.min(eye_region[:,1])
        max_y=np.max(eye_region[:,1])

        gray_eye=eye[min_y:max_y,min_x:max_x]
        _,threshold_eye=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)


        #cut the left eye in half and count how much of the iris is going in the left
        height,width=threshold_eye.shape
        left_side_threshold=threshold_eye[0:height,0:int(width/2)]
        left_side_white=cv2.countNonZero(left_side_threshold)
        #cut the right eye in half and count how much of the iris is going in the right side
        right_side_threshold=threshold_eye[0:height,int(width/2):width]
        right_side_white=cv2.countNonZero(right_side_threshold)

        #in case there is no ratio of white for one of the eyes and to not divide by 0 or divide 0
        if left_side_white==0:
            gaze_ratio=0.5
        elif right_side_white==0:
            gaze_ratio=2
        else:
            gaze_ratio=left_side_white/right_side_white

        return gaze_ratio
#timer that will manage how much time the blinking message stays on the screen
def blink_start():
    global clock
    clock=time.time()

clock=0
while True:
    _,frame=cap.read()

    #gray frame for eficiency
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces=detector(gray)
    for face in faces:
        #for getting a square that shows where your face is
        x,y=face.left(),face.top()
        x1,y1=face.right(),face.bottom()
        cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        landmarks=predictor(gray,face)

        
        #detect blinking
        left_eye_ratio=get_blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio=get_blinking_ratio([42,43,44,45,46,47],landmarks)
        blinking_ratio=(left_eye_ratio+right_eye_ratio)/2

        #timer for blinking message
        if blinking_ratio>4.7:
            blink_start()
        if clock and (time.time()-clock)<0.5:
            cv2.putText(frame,"BLINKING",(50,200),font,3,(255,0,0))
        
        #Gaze detecting
        gaze_ratio_left=get_gaze_ratio([36,37,38,39,40,41],landmarks)
        gaze_ratio_right=get_gaze_ratio([42,43,44,45,46,47],landmarks)
        gaze_ratio=(gaze_ratio_right+gaze_ratio_left)/2
        #cv2.putText(frame,str(gaze_ratio),(50,150),font,2,(0,0,255),3)

        #intervals for each direction
        if gaze_ratio<=0.85:
            cv2.putText(frame,"RIGHT",(50,100),font,2,(0,0,255),3)
        elif 0.85<gaze_ratio<1.3:
                cv2.putText(frame,"CENTER",(50,100),font,2,(0,0,255),3)
        else:
                    cv2.putText(frame,"LEFT",(50,100),font,2,(0,0,255),3)



    cv2.imshow("Frame",frame)

    #press esc key to finish the program
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()