import tkinter as tk
from PIL import Image, ImageTk
import cv2
from time import sleep

face_cascade = cv2.CascadeClassifier('/home/icurus/project/ai_projects/secrurity-2.0/src/cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/icurus/project/ai_projects/secrurity-2.0/src/cascades/data/haarcascade_eye.xml  ')

class MainWindow:
    def __init__(self, window, cap):

        print(face_cascade)
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 1  # Interval in ms to get the latest frame

        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        # creates a canvas equal to the capture width and the capture height
        self.canvas.grid(row=0, column=0)
        # sets the canvas as the first index position

        # Update image on canvas
        self.update_image()
        # calls the camera to run after setting up the canvas

    def convert_and_print(self):
        # Creates an image an image from an memory buffer stream
        self.image = Image.fromarray(self.image)  # to PIL format
        # self.image = Image.fromarray(self.gray)  # to PIL format

        self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format

        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)

    def update_image(self):
        # Get the latest frame and convert image format
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  # changes the colour format to to RGB
        self.gray = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)  # changes the colour format to grey
        self.faces = face_cascade.detectMultiScale(self.gray) #scaleFactor=1.5,  minNeighbors=5
        print(len(self.faces))
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = self.gray[y:y + h, x:x + w]
            roi_color = self.image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        self.convert_and_print()


if __name__ == "__main__":
    root = tk.Tk()  # creates an instance of the Tk class in tkinter
    # MainWindow(root, cv2.VideoCapture(0))
    run = MainWindow(root, cv2.VideoCapture(0))  # both lines run the class Mainloop
    root.mainloop()  # This is runs the interface ensuring that it does not collapse
