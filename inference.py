import json
from pathlib import Path

import numpy as np
import supervision as sv

from deeplabcut.pose_estimation_pytorch.apis.videos import (
    VideoIterator,
    video_inference,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
    create_superanimal_inference_runners,
)

pose_runner, det_runner, model_cfg = create_superanimal_inference_runners(
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}

bodyparts = list(model_cfg["metadata"]["bodyparts"])

vertex_annotator = sv.VertexAnnotator(radius=4)

dataset_dir = Path("/home/juan/Videos/udmt/sam3_dataset_mice")
output_dir = dataset_dir / "dlc_supervision"
output_dir.mkdir(exist_ok=True)

for video_path in sorted(dataset_dir.iterdir()):
    if video_path.suffix.lower() not in VIDEO_SUFFIXES:
        continue

    # One dict per frame with a "bodyparts" array of shape
    # (n_individuals, n_bodyparts, 3), where the last axis is (x, y, score).
    predictions = video_inference(
        VideoIterator(video_path),
        pose_runner=pose_runner,
        detector_runner=det_runner,
    )

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    frames = sv.get_video_frames_generator(str(video_path))

    annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
    json_path = output_dir / f"{video_path.stem}.json"

    # COCO keypoints dataset: one "image" per frame, one "annotation" per
    # individual. Bodyparts are the single category's keypoints.
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "animal",
                "supercategory": "animal",
                "keypoints": bodyparts,
                "skeleton": [],
            }
        ],
    }
    annotation_id = 1

    with sv.VideoSink(str(annotated_path), video_info) as video_sink:
        for frame_index, (frame, prediction) in enumerate(zip(frames, predictions)):
            pose = np.asarray(prediction["bodyparts"], dtype=np.float32)
            xy = pose[..., :2]
            keypoint_confidence = pose[..., 2]

            image_id = frame_index + 1
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": f"{video_path.stem}_frame_{frame_index:06d}.png",
                    "width": video_info.width,
                    "height": video_info.height,
                    "video": video_path.name,
                    "frame_index": frame_index,
                }
            )

            # Detector boxes are already COCO [x, y, width, height].
            bboxes = np.asarray(prediction["bboxes"], dtype=np.float32)
            bbox_scores = np.asarray(prediction["bbox_scores"], dtype=np.float32).reshape(-1)

            for individual_xy, individual_conf, bbox, score in zip(xy, keypoint_confidence, bboxes, bbox_scores):
                # COCO visibility flag: 2 = labeled/visible, 0 = not labeled.
                # NaN predictions become (0, 0, 0), matching COCO's missing point.
                visible = np.isfinite(individual_xy).all(axis=-1)
                keypoints = np.zeros((len(bodyparts), 3), dtype=np.float32)
                keypoints[visible, :2] = individual_xy[visible]
                keypoints[visible, 2] = 2

                x, y, w, h = bbox.tolist()
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": float(w * h),
                        "iscrowd": 0,
                        "num_keypoints": int(visible.sum()),
                        "keypoints": keypoints.reshape(-1).tolist(),
                        "score": float(score),
                        "keypoint_scores": individual_conf.tolist(),
                    }
                )
                annotation_id += 1

            # NaNs are zeroed so the annotator (which skips (0, 0) points and
            # cannot cast NaN to int) treats them as missing.
            key_points = sv.KeyPoints(
                xy=np.nan_to_num(xy, nan=0.0),
                keypoint_confidence=np.nan_to_num(keypoint_confidence, nan=0.0),
                visible=np.isfinite(xy).all(axis=-1),
            )
            annotated = vertex_annotator.annotate(frame.copy(), key_points)
            video_sink.write_frame(annotated)

    with open(json_path, "w") as f:
        json.dump(coco, f)

    print(f"Saved annotated video -> {annotated_path}")
    print(f"Saved COCO output     -> {json_path}")
