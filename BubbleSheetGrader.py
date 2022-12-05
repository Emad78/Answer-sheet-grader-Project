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


bsg = BubbleSheetGrader('data/camScanedSample/sample (4).png')
bsg.pre_process()
