import argparse
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
# import pickle
from time import sleep


# face_cascade = cv2.CascadeClassifier('/home/icurus/project/ai_projects/secrurity-2.0/src/cascades/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/home/icurus/project/ai_projects/secrurity-2.0/src/cascades/data/haarcascade_eye.xml')
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("./src/recognizers/face-trainner.yml")
#
# labels = {"person_name": 0}
# with open("pickles/face-labels.pickle", 'rb') as f:
#     og_labels = pickle.load(f)
#     labels = {v: k for k, v in og_labels.items()}
#     print(labels)


class MainWindow:
    def __init__(self, window, cap, args):

        # print(face_cascade)
        self.window = window
        self.cap = cap
        self.args = args
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 1  # Interval in ms to get the latest frame

        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        # self.canvas = tk.Canvas(self.window)
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

    def decode_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

    def update_image(self):
        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(self.args["east"])
        vs = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        (H, W) = vs.shape[:2]
        rW = W / float(self.width)
        rH = H / float(self.height)
        blob = cv2.dnn.blobFromImage(vs, 1.0, swapRB=True, crop=False)
        net.setInput(blob)
        (self.scores, self.geometry) = net.forward(layerNames)
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = self.decode_predictions(self.scores, self.geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # draw the bounding box on the frame
            cv2.rectangle(vs, (startX, startY), (endX, endY), (0, 255, 0), 2)
            self.convert_and_print()


if __name__ == "__main__":
    root = tk.Tk()  # creates an instance of the Tk class in tkinter
    # MainWindow(root, cv2.VideoCapture(0))
    ap = argparse.ArgumentParser()
    ap.add_argument("-east", "--east", type=str, default="frozen_east_text_detection.pb",
                    help="path to input EAST text detector")
    ap.add_argument("-v", "--video", type=str, default="0",
                    help="path to optinal input video file")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
                    help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
                    help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())
    run = MainWindow(root, cv2.VideoCapture(0),args)  # both lines run the class Mainloop
    root.mainloop()  # This is runs the interface ensuring that it does not collapse
