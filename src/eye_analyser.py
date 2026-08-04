import numpy as np

class EyeAnalyser:
    LEFT_EYE= [33,160,158,133,153,144]
    RIGHT_EYE= [362,385,387,263,373,380]

    EAR_THRESHOLD= 0.22

    def __init__(self, threshold= EAR_THRESHOLD):
        self.threshold = threshold

    def _distance(self,p1,p2):
        return np.linalg.norm(p1-p2)

    def _compute_ear(self, eye_points):
        p1,p2,p3,p4,p5,p6 = eye_points
        vertical_1 = self._distance(p2,p6)
        vertical_2 = self._distance(p3,p5)
        horizontal = self._distance(p1,p4)

        if horizontal == 0:
            return 0.0
        ear = (vertical_1 + vertical_2)/(2.0 * horizontal)
        return ear

    def analyse(self, landmark_points):
        if landmark_points is None:
            return None, None, None

        left_eye = landmark_points[self.LEFT_EYE]
        right_eye= landmark_points[self.RIGHT_EYE]

        left_ear= self._compute_ear(left_eye)
        right_ear= self._compute_ear(right_eye)

        average_ear= (left_ear + right_ear) / 2

        if average_ear > self.threshold:
            eye_state= "Open"
        else:
            eye_state = "CLOSED"

        return eye_state, left_ear, right_ear