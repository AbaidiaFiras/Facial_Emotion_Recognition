#Loading librairies:
from keras.models import load_model
import  cv2
from google.colab.patches import cv2_imshow

#Loading the model weights and the haarcascade detector:
model = load_model('/content/drive/MyDrive/facial /FER13.h5')
face = cv2.CascadeClassifier('/content/drive/MyDrive/facial /haarcascade_frontalface_alt.xml')

#Read the image and feed it to Haarcascade:
path="//content/dataset/train/fear/Training_10208260.jpg"
image= cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
for (x,y,w,h) in faces:
  cv2.rectangle(gray, (x,y) , (x+w,y+h) , (255,255,0) , 1 )
cv2_imshow(gray)

#Crop the detected face:
cropped =gray[y:y+h,x:x+w]
cv2_imshow(gray)

#Classify the cropped face (whether it's happy , angry , ...):
#Resize the cropped image before feeding it to the model:
gray = cv2.resize(gray,(48,48))

#Normalize the cropped image also:
gray= gray/255
gray= gray.reshape(48,48,-1)
gray = np.expand_dims(gray,axis=0)
prediction_result=model.predict(gray)

#Predict the class and display it:
prediction_result=model.predict(gray)
prediction_result=np.argmax(prediction_result,axis=1)
prediction_result

#Matching the class with the label:
if(prediction_result[0]==0):
    lbl='Angry'
if(prediction_result[0]==1):
    lbl='disgust'
if(prediction_result[0]==2):
    lbl='fear'
if(prediction_result[0]==3):
    lbl='happy'
if(prediction_result[0]==4):
    lbl='neutral'
if(prediction_result[0]==5):
    lbl='sad'
if(prediction_result[0]==6):
    lbl='surprise'
  
#The result:
lbl
