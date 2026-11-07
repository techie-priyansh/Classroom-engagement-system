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

`seat_identity.py` recognises the same student when the camera returns to a region: identity is tied to the
seat. It aligns every visit to the first visit of its region (`seat_map.py`), finds the seats from the head
positions (`head_x`, `head_y` in the CSV) and adds a `seat_id` column. `compare_visits.py` measures camera drift
between two visits. With a `seat_id` column, `engagement_score.py` draws one line per seat across all visits
(`seat_timeline.png`).

`person_identity.py` combines the seat with face recognition (insightface buffalo_l, embeddings from small crops around
each head, only frames with a frontal enough face). Tracks of one region are grouped into persons: a face match
links them, the seat adds a bonus when it agrees and a penalty when it differs, a track without a usable face is
linked by seat alone, and tracks visible at the same time are never the same person. It adds a `person_id` column
(`R<region>-P<n>`) and a `persons.csv` report; `engagement_score.py` then uses `person_id` instead of `seat_id`.
On the 2-minute test clip it gave 8 / 8 / 3 persons, the same as seats alone; seats alone and faces alone each give
the same people, so the combination matters when one of the two fails (student moved, head turned away).

## Run

From the repository root:

```bash
python src/video_pipeline.py                                   # video -> CSV + annotated video
python src/eye_features.py   results/video_test/student_data1.csv
python src/seat_identity.py <video> results/video_test/student_data1.csv   # adds seat_id, writes check images
python src/engagement_score.py results/video_test/student_data1_seats.csv   # seat identity only
python src/person_identity.py <video> results/video_test/student_data1.csv   # seat + face -> student_data1_ids.csv
python src/engagement_score.py results/video_test/student_data1_ids.csv      # uses person_id
python src/validate_head_pose.py                               # head pose quality report
python src/compare_visits.py clipA.mp4 8.0 clipB.mp4 8.0       # camera drift between two visits
```

## CSV columns (`student_data1.csv`)

`frame, time, region_id, student_id, yaw, pitch, roll, left_ear, right_ear, avg_ear, gaze,
shoulder_tilt, head_drop, head_offset, hand_raised, hands_near_face, head_x, head_y`
(`seat_identity.py` adds `seat_id`, `person_identity.py` adds `seat_id` and `person_id`).

Only settled frames are written. `shoulder_tilt` is in degrees; `head_drop` and
`head_offset` are in units of shoulder width. Empty values mean "not measured", never "zero".

## Engagement score

Written as `student_scores.csv` (per visit), `student_seconds.csv` (per second), `class_seconds.csv` and two graphs
(`student_timeline.png`, `student_seconds_timeline.png` with student IDs on the lines, and `student_seconds_grid.png`, one panel per student, for short clips). Built from head direction (front / down / turned away), eye closure (PERCLOS, long closures) and a raised
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
python test_seat_identity.py
python test_person_identity.py
python test_head_point.py
```

## Known limitations

- Tracker IDs restart at every visit; run `person_identity.py` (seat + face) to link students across visits.
  With `seat_identity.py` alone, a student who changes seat is a new seat.
- Face matching was measured on one 2-minute clip (8 students per region, faces about 60 px wide); re-check the
  threshold `FACE_MERGE` with a bigger class. Reading faces on CPU takes about 3.5x the clip length; for an hour
  it has to be run in parallel (per visit) or on a GPU.
- Head-down students (writing) cannot be separated from bored ones with a front camera.
- The sweep log of the recorder was tested only with a fake camera controller.
- No ground truth yet, so the score is not validated.
