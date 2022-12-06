import cv2

class BubbleSheetGrader:
    """Class for reading image with BubbleSheet answer and grading that."""
    def __init__(self, path):
        """Load and resize image"""
        self.image = cv2.imread(path)
        self.image = cv2.resize(self.image, (780, 540))
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
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(self.image, contours, -1, (0, 0, 0), 2)
        self.show_image(self.image)
        for i in range(len(contours)):
            cp = self.image.copy()
            print(cv2.contourArea(contours[i]))
            cv2.drawContours(cp, contours[:i+1], -1, (255, 0, 0), 2)
            self.show_image(cp)
            
    def show_image(self, img):
        cv2.imshow("show", img)
        cv2.waitKey(0)
        

bsg = BubbleSheetGrader('data/standardSample/sample (2).bmp')
bsg.pre_process()
bsg.detect_answer_box()
