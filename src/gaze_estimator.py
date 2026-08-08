import numpy as np





class GazeEstimator:
    LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144,]

    RIGHT_EYE = [
        362,
        385,
        387,
        263,
        373,
        380,
    ]
    def __init__(self):
        self.horizontal_threshold = 20.0
        self.vertical_threshold = 20.0

    def _eye_center(self, eye_points):
        return np.mean(eye_points, axis=0)
    
    def estimate(
            self,
            landmark_points,
            head_pose,
    ):
        """
        Parameters: 
        landmark_ponts: ndarray (478,2)
        head_pose: (ywa,pitch, roll)
        returns
        gaze_direction: str
        """
        if landmark_points is None:
            return None
        if head_pose is None:
            return None

        yaw, pitch, roll = head_pose

        left_eye = landmark_points[self.LEFT_EYE]
        right_eye = landmark_points[self.RIGHT_EYE]

        left_center= self._eye_center(left_eye)
        right_center= self._eye_center(right_eye)
        eye_center = (left_center + right_center) / 2

        #Horizontal gaze

        if yaw > self.horizontal_threshold:
            horizontal = "LEFT"
        elif yaw < -self.horizontal_threshold:
            horizontal = "RIGHT"
        else:
            horizontal= "FORWARD"

        #vertical gaze

        if pitch > self.vertical_threshold:
            vertical = "DOWN"

        elif pitch < -self.vertical_threshold:
            vertical = "UP"
        else:
            vertical = "CENTER"

        #Final gaze
        if horizontal != "FORWARD":
            return horizontal

        if vertical != "CENTER":
            return vertical

        return "FORWARD"

