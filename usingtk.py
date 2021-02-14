import tkinter as tk
from PIL import Image, ImageTk
import cv2


class MainWindow:
    def __init__(self, window, cap):
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

    def update_image(self):
        # Get the latest frame and convert image format
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  # changes the colour format to to RGB
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2GRAY)   # changes the colour format to grey
        # Creates an image an image from an memory buffer sream
        self.image = Image.fromarray(self.image)  # to PIL format

        self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format

        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)


if __name__ == "__main__":
    root = tk.Tk()  # creates an instance of the Tk class in tkinter
    # MainWindow(root, cv2.VideoCapture(0))
    run = MainWindow(root, cv2.VideoCapture(0))  # both lines run the class Mainloop
    root.mainloop()  # This is runs the interface ensuring that it does not collapse
