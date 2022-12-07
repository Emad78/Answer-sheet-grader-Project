import cv2
import pickle
import numpy as np

class BubbleSheetGrader:
    """Class for reading image with BubbleSheet answer and grading that."""
    def __init__(self, path):
        """Load and resize image"""
        self.image = cv2.imread(path)
        self.image = cv2.resize(self.image, (780, 540))
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
        de_noised = cv2.fastNlMeansDenoising(gray, None, 2, 5, 10)
        blurred = cv2.GaussianBlur(de_noised,(3, 3), 0)
        self.pre_processed_image = blurred

    def detect_answer_sheet(self):
        """Detect answer sheet in image"""
    
    def detect_answer_box(self):
        """Detect answer box in answer sheet"""
        edged = cv2.Canny(self.pre_processed_image, 10, 100)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        box_contours = list(filter(lambda x: cv2.contourArea(x)>500, contours))
        box_contours = sorted(box_contours, key=lambda x: cv2.contourArea(x), reverse=True)
        answer_box = self.crop_contour_area(box_contours[0], self.image)
        self.show_image(answer_box)
        return answer_box
    
    def crop_contour_area(self, contour, image):
        cropped_contours = []
        x,y,w,h= cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]
    def show_image(self, img):
        cv2.imshow("show", img)
        cv2.waitKey(0)
        

bsg = BubbleSheetGrader('data/standardSample/sample (7).bmp')
bsg.pre_process()
bsg.detect_answer_box()
