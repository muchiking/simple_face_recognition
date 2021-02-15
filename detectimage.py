import  cv2
classifier = cv2.CascadeClassifier("/home/icurus/project/ai_projects/secrurity-2.0/src/cascades/data/haarcascade_frontalface_default.xml")
img = cv2.imread('mc.jpg')
face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(face)
print(type(faces), faces)
for (x, y, w, h) in faces:
  img = cv2.imwrite("facesa.png", cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3))