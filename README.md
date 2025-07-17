# Automated Face Mosaic

This project automatically detects faces in video footage using a combination of object detection and face recognition technologies. It performs real-time face tracking and applies mosaic (pixelation) over detected faces for privacy protection.

The system uses:

- **YOLO** for fast and accurate face detection
- **ArcFace** for robust face feature embedding and identity tracking
- **SORT** (Simple Online and Realtime Tracking) for temporal tracking across video frames
- **OpenCV** for video frame handling and mosaic effect rendering



## config.yaml

| Field            | Description                                                | Example                        |
|------------------|------------------------------------------------------------|--------------------------------|
| `input_path`     | Path to the input video file to process.                   | `"src/video/test121.mp4"`      |
| `output_path`    | Path where the output video with mosaic will be saved.     | `"src/output/output.mp4"`      |


| Field            | Description                                                      | Example               |
|------------------|------------------------------------------------------------------|-----------------------|
| `image_size`     | Image input size for YOLO model during training.                | `640`                 |
| `batch_size`     | Number of samples per batch during training.                    | `32`                  |
| `epoch`          | Total number of training epochs.                                 | `50`                  |
| `export_path`    | Directory path to save the trained YOLO model.                  | `".src/model"`        |
