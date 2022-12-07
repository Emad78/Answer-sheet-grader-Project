import cv2
import pickle
import numpy as np
import imutils

class BubbleSheetGrader:
    """Class for reading image with BubbleSheet answer and grading that."""
    def __init__(self, path):
        """Load and resize image"""
        self.image = cv2.imread(path)
        self.image = imutils.resize(self.image, height = 1000)
        self.box_template = []
        with open('data/template/rectangle.temp', 'rb') as f:
           self.box_template.append(pickle.load(f))
        with open('data/template/square.temp', 'rb') as f:
           self.box_template.append(pickle.load(f))
        with open('data/template/bubble.temp', 'rb') as f:
           self.bubble = pickle.load(f)
           
           
        self.pre_processed_image = None

    def pre_process(self):
        """Pre processing image"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.pre_processed_image = blurred

    def detect_answer_sheet(self):
        """Detect answer sheet in image"""
    
    def detect_answer_box(self):
        """Detect answer box in answer sheet"""
        edged = cv2.Canny(self.pre_processed_image, 75, 200)
        contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        screenCnt = []

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.002 * peri, True)
            if len(approx) == 4:
                screenCnt.append(approx)
        
        answer_box = self.crop_contour_area(screenCnt[0], self.pre_processed_image)
        return answer_box
    
    def crop_contour_area(self, contour, image):
        """Crop contour area from input image"""
        x,y,w,h= cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]
    
    def show_image(self, img):
        """Using for show results"""
        cv2.imshow("show", img)
        cv2.waitKey(0)
        

bsg = BubbleSheetGrader('data/standardSample/sample (7).bmp')
bsg.pre_process()
bsg.show_image(bsg.detect_answer_box())
