import cv2, os, time, imutils, imghdr
from pygoogle_image import image as pi

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'Datasets'

## If Datasets folder not exist, create that folder
if not os.path.isdir(datasets):
	os.mkdir(datasets)
	
personName = input("Enter the person name : ")
imagesCount = input("No of images need : ")

path = os.path.join(datasets, personName)
if not os.path.isdir(path):
	os.mkdir(path)

##Download image from Google
pi.download(personName, limit=int(imagesCount))
downloaded_path = os.path.join("images", personName)
downloaded_path = downloaded_path.replace(" ", "_")

(width,height) = (300,300)

## Face Detection and save the faces into folder
face_cascade = cv2.CascadeClassifier(haar_file)
count = 1

for filename in os.listdir(downloaded_path):
	print(count)
	im = cv2.imread(os.path.join(downloaded_path, filename))

	if imghdr.what(os.path.join(downloaded_path, filename))=='jpeg':
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		
		faces =  face_cascade.detectMultiScale(gray, 1.3,4)
		
		for (x,y,w,h) in faces:
			cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
			face = gray[y:y + h, x:x + w]
			face_resize = cv2.resize(face, (width, height))
			cv2.imwrite('%s/%s.png' % (path,count), face_resize)
		count += 1
		
		cv2.imshow(filename, imutils.resize(im, width=500))
		key = cv2.waitKey(1000)
		if key == 27:
			break

cv2.destroyAllWindows()