import numpy as np

class PostureAnalyser:
    def _angle(self,a,b,c):
        """
        Angle ABC in degrees.
        """
        a= np.array(a, dtype= np.float64)
        b= np.array(b, dtype= np.float64)
        c= np.array(c, dtype= np.float64)

        ba = a - b
        bc = c - b

        denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denominator ==0:
            return None
        cosine = np.dot(ba,bc)/denominator
        cosine = np.clip(cosine,-1.0,1.0)
        return np.degrees(np.arccos(cosine))

    def analyse(self, keypoints):
        if keypoints is None:
            return None

        left_shoulder = keypoints["left_shoulder"]["point"]
        right_shoulder = keypoints["right_shoulder"]["point"]


        left_hip = keypoints["left_hip"]["point"]
        right_hip = keypoints["right_hip"]["point"]

        left_knee = keypoints["left_knee"]["point"]
        right_knee = keypoints["right_knee"]["point"]

        shoulder_center = (left_shoulder + right_shoulder)/2
        hip_center = (left_hip + right_hip)/2

        #####Torso angle###

        torso_vector = shoulder_center - hip_center
        torso_angle = np.degrees(
            np.arctan2(
                abs(torso_vector[0]),
                abs(torso_vector[1])
            )
        )

        #knee angles#
        left_knee_angle= self._angle(
            left_hip,
            left_knee,
            keypoints["left_ankle"]["point"]
        )

        right_knee_angle= self._angle(
            right_hip,
            right_knee,
            keypoints["left_ankle"]["point"]
        )

        return {
            "torso_angle" : torso_angle,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
        }
