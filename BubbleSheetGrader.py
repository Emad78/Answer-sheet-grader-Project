import pickle
import cv2
import imutils
from imutils import contours as cont
import numpy as np

class BubbleSheetGrader:
    """Class for reading image with BubbleSheet answer and grading that."""

    def __init__(self, path):
        """Load and resize image"""

        self.image = cv2.imread(path)
        self.image = imutils.resize(self.image, height = 1000)
        self.box_template = []

        with open('data/template/rectangle.temp', 'rb') as file:
            self.box_template.append(pickle.load(file))
        with open('data/template/square.temp', 'rb') as file:
            self.box_template.append(pickle.load(file))
        with open('data/template/bubble.temp', 'rb') as file:
            self.bubble = pickle.load(file)

        self.pre_processed_image = None
        
        self.correct_answers = [1, 1, 2, 1, 2, 1, 1, 2, 1, 4]

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
        screen_contours = []

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.002 * peri, True)
            if len(approx) == 4:
                screen_contours.append(approx)

        answer_box = self.crop_contour_area(screen_contours[0], self.pre_processed_image)
        return answer_box

    def detect_bubbles(self, answer_box):
        """Detecting bubbles in answer box"""

        binary_image = cv2.threshold(answer_box, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        self.show_image(binary_image)
        contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        area = list(map(cv2.contourArea, contours))
        area_average = sum(area) / len(area)
        contours = list(filter(lambda x: cv2.contourArea(x) > area_average, contours))

        bubbles_contours = []
        for contour in contours:
            (_, _, w, h) = cv2.boundingRect(contour)
            ar = w / float(h)
            if 0.9 <= ar <= 1.10:
                bubbles_contours.append(contour)

        bubbles_contours = cont.sort_contours(bubbles_contours, method="left-to-right")[0]
        bubbles_contours = cont.sort_contours(bubbles_contours, method="top-to-bottom")[0]
        self.extract_marked_bubbles(answer_box, bubbles_contours)
        return bubbles_contours

    def extract_marked_bubbles(self, binary_box, bubbles):
        """Extract marked and unmarked bubbles per rows"""

        marked = []
        row = []
        last_x = 0
        for bubble in bubbles:
            x, _, _, _= cv2.boundingRect(bubble)
            if x < last_x:
                marked.append(row)
                row = []
            last_x = x
            mask = np.zeros(binary_box.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            mask = cv2.bitwise_and(binary_box, binary_box, mask=mask)
            total = cv2.countNonZero(mask)
            if total > (1.08*cv2.contourArea(bubble)):
                row.append(0)
            else:
                row.append(1)
        self.find_answer_per_question(marked)
        return marked

    def find_answer_per_question(self, marked):
        """Find answer per question"""
        row_number = len(marked)
        answers = [[] for i in range(len(marked[0])//4)]
        for row in range(row_number):
            for col, _ in enumerate((answers)):
                answer = [marked[row][4*col], marked[row][4*col+1], marked[row][4*col+2], marked[row][4*col+3]]
                answers[col].append(answer)

        final_answers = []
        for answer in answers:
            final_answers += answer
        self.grading_answers(final_answers)
        return final_answers

    def grading_answers(self, answers):
        """Grading each answers"""
        result = []
        for i, answer in enumerate(answers):
            if i >= len(self.correct_answers):
                break
            if sum(answer) > 1:
                result.append(-1)
            elif sum(answer) == 0:
                result.append(0)
            elif answer.index(1) + 1 == self.correct_answers[i]:
                result.append(1)
            else:
                result.append(-1)
        print(result)
        return result

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
bsg.detect_bubbles(bsg.detect_answer_box())
