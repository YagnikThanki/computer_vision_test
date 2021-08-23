import cv2
import numpy as np 

haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

test1 = cv2.imread('data/test1.jpg')

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
	img_copy = np.copy(colored_img)
	#convert the test image to gray image as opencv face detector expects gray images
	gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
	
	#let's detect multiscale (some images may be closer to camera than others) images
	faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
	
	#go over list of faces and draw them as rectangles on original colored img
	for (x, y, w, h) in faces:
		cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
	return img_copy

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	print("press \"c\" to save, press \"q\" to quit ")
	while True:
		ret, frame = cap.read()
		face_detected_frame = detect_faces(haar_face_cascade, frame)
		# processed_image_fast_means = cv2.fastNlMeansDenoisingColored(face_detected_frame,None,10,10,7,21)    # method 2 
		# print(face_detected_frame.dty)
		processed_image_fast_means = cv2.fastNlMeansDenoisingColored(face_detected_frame,None,10,10,7,21)    # method 2 
		processed_image_gaussian = cv2.GaussianBlur(face_detected_frame, (9,9),0) ## method 1 
		cv2.imshow("orignal", frame)
		cv2.imshow("processed_image_fast_means", processed_image_fast_means)
		cv2.imshow("processed_image_gaussian", processed_image_gaussian)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		if key & 0xFF == ord('c'):
			cv2.imwrite("orignal_frame.jpg", frame)
			cv2.imwrite("processed_gaussian_frame.jpg", processed_image_gaussian)
			cv2.imwrite("processed_fastmeans_frame.jpg", processed_image_fast_means)

	cap.release()