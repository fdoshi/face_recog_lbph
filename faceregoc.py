import cv2
import numpy as np
import pygame

print 'started'
model=cv2.createLBPHFaceRecognizer(radius=1,neighbors=9,grid_x=8,grid_y=8,threshold=60)
model.load("./trainer.xml")
print "loaded"

#initialize pygame
pygame.init()
disp=pygame.display.set_mode((640,480))
disp.fill((0,0,255))
#set font for displaying the information
font=pygame.font.SysFont('Calibri',76,True,False)

###################################################################################################################################
def recognize_face():
    #webcam initializtion
    cap=cv2.VideoCapture(0)
    face_cas=cv2.CascadeClassifier("./haarcascade_frontalface_alt_tree.xml")
    running = True

    #set people
    people=['rohan','rohit','aproocv','riya','director','aksha','']

    #count for the images that are recognised// total_image for total images detected
    count=[0,0,0,0,0,0,0,0,0]
    total_image=0

    #if the person in range, simulation right now
    in_range=True


    while(cap.isOpened and running and total_image<=25 and in_range):   ### WHEN IMPLIMENTING ON PIE
        r,fram=cap.read()

        color_frame=fram
        color_frame=cv2.cvtColor(color_frame,cv2.COLOR_BGR2RGB)

        img=pygame.surfarray.make_surface(color_frame)
        img=pygame.transform.rotate(img,-90)

        fram=cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY)
        fram=cv2.flip(fram,1)

        faces=face_cas.detectMultiScale(fram)
        test=fram
        detect_name=""
        for (x,y,w,h) in faces:
            total_image+=1
            cv2.rectangle(color_frame,(x-30,y-30),(x+w+30,y+h+30),(255,255,255))
            test=fram[y+30:y+h-15,x+10:x+w-10]
            pygame.draw.line(img,(255,0,0),(x,0),(x,480))
            pygame.draw.line(img,(255,0,0),(x+w,0),(x+w,480))
            pygame.draw.line(img,(255,0,0),(0,y),(640,y))
            pygame.draw.line(img,(255,0,0),(0,y+h),(640,y+h))
    #        r = 112.0 / test.shape[1]
            dim = (300, 300)
            confidence=10000

            try:
                test = cv2.resize(test, dim, interpolation = cv2.INTER_AREA)
                num,confidence=model.predict(np.asarray(test,dtype=np.uint8))
                confidence=np.round(confidence)
                # print model.predict(np.asarray(test,dtype=np.uint8))
                # cv2.imshow("detect",np.array(test))
            except: num=-1
            #
            # if(confidence>12000):
            #     num=-1
            if(num>=0 and num<len(people)):
                count[num]+=1
                detect_name=people[num]+":"+str(confidence)+"%"
                print detect_name
            else:
                detect_name="unknown"

            target_font=font.render(detect_name,True,(0,0,0))
            img.blit(target_font,(0,0))
            # cv2.putText(color_frame,detect_name,(x,y),cv2.FONT_ITALIC,w*0.005,(255,255,255))
        # cv2.imshow("bla",color_frame)


        disp.blit(img,(0,0))
        pygame.display.update()


        for events in pygame.event.get():
            if events.type==pygame.QUIT:
                running=False
            if events.type==pygame.KEYDOWN:
                if events.key==pygame.K_ESCAPE:
                    running=False

    print count
    print total_image
    maxi=0

    cap.release()
    for _ in range(len(count)): ##CHECK WHICH PERSON HAS MAX VAULE
        if count[_]>count[maxi]:
            maxi=_

    print people[maxi] #DISPLAY THE PERSON
    if(((float(count[maxi]))/(float(total_image)))>0.6) and running: #IF THE PERSON HAVIGN MAXIMUM VALUE IF DETECTED MORE THAN 60%
        # print 'open'
        return True
        #stop for a while

    else:
        print "not detected"
        return False
        #repeat with total and count =0
    cap.release()



