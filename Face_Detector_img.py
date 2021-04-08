import cv2

# pre trained data 
#classifier is to classify smthg by face 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose img to detect face
#import img 
img = cv2.imread('groupe.png')

#convert it to grey ! 
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces whatever the scale is 
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#rectangle around the face
#2 is how thick is the rectangle
#0,255,0 is the color of the rectangle
for (x,y,w,h) in face_coordinates:
    #(x,y,w,h) = face_coordinates[0]  #0 is for the first face detected
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

#p display the img
cv2.imshow('detector', img)
#without this, the pic will show up for few ms and then disappear
cv2.waitKey()


