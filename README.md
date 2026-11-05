# Classroom-engagement-system

Software that estimates a per-student engagement score in a classroom from video.

**Setup 1 (current):** one PTZ camera (CM-Z322-RP, ONVIF) sweeps three regions of the class in a loop
(about 8 s recorded per region, about 2 s moving). Only the frames where the camera stands still and
in focus are analysed, and each student gets one engagement score per visit.

**Setup 2 (future, on hold):** several synchronised cameras with homography.

## Install

```bash
python -m venv mtp-env && source mtp-env/bin/activate
pip install -r requirements.txt
```

The YOLO weights (`yolo11s.pt`, `yolo11n-pose.pt`) are downloaded by ultralytics on first use.

## Paths

No script uses a path from one computer. `src/paths.py` finds the repository folder itself:

| Name | Default |
|---|---|
| `PROJECT_ROOT` | the repository folder |
| `DATASET_DIR` | `PROJECT_ROOT/dataset` (PTZ recordings) |
| `CALIBRATION_DIR` | `calibration/` next to the repository, or the folder in the environment variable `MTP_CALIBRATION_DIR` |

## Pipeline

```
ptz_record_sweep.py     record the Region 1 -> 2 -> 3 sweep (+ <video>.sweep.json log)
        |
settle.py               find the frames where the camera is still and in focus (the "visits")
        |
video_pipeline.py       per settled frame:
        |                 body_pose.py      YOLO pose + ByteTrack (one model, USE_POSE_ONLY = True)
        |                 face_mesh.py      MediaPipe 478 landmarks
        |                 head_pose.py      yaw / pitch / roll (PnP)
        |                 eye_analyser.py   EAR
        |                 gaze_estimator.py LEFT / RIGHT / DOWN / FORWARD
        |                 posture_analyser.py  shoulder tilt, head drop, hands
        v
results/video_test/student_data1.csv
        |
eye_features.py         per student and visit: baseline EAR, PERCLOS, blinks, long closures
engagement_score.py     score 0-100 per student per visit AND per second, class means, graphs
```

`seat_map.py` and `compare_visits.py` are a prototype for recognising the same student when the camera
returns to a region (identity tied to the seat). They have not been tested across real visits yet.

## Run

From the repository root:

```bash
python src/video_pipeline.py                                   # video -> CSV + annotated video
python src/eye_features.py   results/video_test/student_data1.csv
python src/engagement_score.py results/video_test/student_data1.csv
python src/validate_head_pose.py                               # head pose quality report
python src/compare_visits.py clipA.mp4 8.0 clipB.mp4 8.0       # camera drift between two visits
```

## CSV columns (`student_data1.csv`)

`frame, time, region_id, student_id, yaw, pitch, roll, left_ear, right_ear, avg_ear, gaze,
shoulder_tilt, head_drop, head_offset, hand_raised, hands_near_face`

Only settled frames are written. `shoulder_tilt` is in degrees; `head_drop` and
`head_offset` are in units of shoulder width. Empty values mean "not measured", never "zero".

## Engagement score

Written as `student_scores.csv` (per visit), `student_seconds.csv` (per second), `class_seconds.csv` and two graphs
(`student_timeline.png`, `student_seconds_timeline.png`). Built from head direction (front / down / turned away), eye closure (PERCLOS, long closures) and a raised
hand, over the frames where the head direction was measured. If fewer than half of the frames were
measured the result is "unknown", not "low". All weights are in `CONFIG` at the top of
`src/engagement_score.py`.

The weights and the High / Medium / Low limits are **starting values, not calibrated**. "Front" is assumed
to be the direction of the camera, which is right only if the camera is next to the board.

## Tests

No camera needed:

```bash
cd src
python test_posture_analyser.py
python test_engagement_score.py
python test_student_manager.py
```

## Known limitations

- Student IDs restart at every visit; scores of the same student cannot be linked across visits until the
  seat map is used.
- Head-down students (writing) cannot be separated from bored ones with a front camera.
- The sweep log of the recorder was tested only with a fake camera controller.
- No ground truth yet, so the score is not validated.
