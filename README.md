# Deepstream-Box: DBox, Out-of-the-box AI deployment tool

Why not make the deployment process simpler and more efficient?

For a long time, the author has been committed to developing a more efficient deployment platform, and DeepStreamX has emerged as the times require against this backdrop.

- **Clarity**:  Configure the `.txt` files in the config folder, and you can run the program with one click.
- **Scalability**: Follow the operation guidelines, and you can quickly deploy custom models.
- **Efficiency**: Allocate modules such as decoding and inference properly to enable more efficient deployment.
###

## Update:

- `16 Jan 2026`. [YOLO26](https://github.com/ultralytics/ultralytics). Done
- `13 Jan 2026`. [YOLO-Master](https://github.com/isLinXu/YOLO-Master). Done
- `9 Jan 2026`. [Fast-SAM](https://github.com/CASIA-LMC-Lab/FastSAM). Done
- `7 Jan 2026`. [Swim-transformer](https://github.com/microsoft/Swin-Transformer). Done

###

## Dependence

1. Ubantu = 22.04 (PC or Jetson)
2. TensorRT >= 8.6.1
3. CUDA >= 12.2
4. DeepStream >= 7.0
5. GStreamer = 1.20.3

## How to use

Please refer to the `docs` folder; each tutorial is divided into five detailed steps.

* [Preparatory Work](#preparatory-work)
* [Export Onnx](#export-onnx)
* [Build Engine](#build-engine)
* [Compile Target](#compile-target)
* [Edit the config file](#edit-the-config-file)
* [Running the program](#testing-the-model)

## Baseline

| Baseline | model | task | tutorial | quantize | onnx | engine |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [YOLO26](./docs) | yolo26n.pt | det | ✅ | ✅ | [yolo26n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo26n.onnx) | [yolo26n-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-det_b1_int8.engine)<br>[yolo26n-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-det_b32_int8.engine) |
|  | yolo26n-cls.pt | cls | ✅ | ✅ | [yolo26n-cls.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo26n-cls.onnx) | [yolo26n-cls_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-cls_b1_int8.engine)<br>[yolo26n-cls_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-cls_b32_int8.engine) |
|  | yolo26n-seg.pt | seg | ✅ | ✅ | [yolo26n-seg.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo26n-seg.onnx) | [yolo26n-seg_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-seg_b1_int8.engine)<br>[yolo26n-seg_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-seg_b32_int8.engine) |
|  | yolo26n-pose.pt | pose | ✅ | ✅ | [yolo26n-pose.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo26n-pose.onnx) | [yolo26n-pose_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-pose_b1_int8.engine)<br>[yolo26n-pose_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo26n-pose_b32_int8.engine) |
| [YOLO-Master](./docs) | yolo-master-esmoe-n.pt | det | ✅ | ✅ | [yolo-master-esmoe-n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo-master-esmoe-n.onnx) | [yolo-master-esmoe-n_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo-master-esmoe-n_b1_int8.engine)<br>[yolo-master-esmoe-n_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo-master-esmoe-n_b32_int8.engine) |
|  | yolo-master-seg-n.pt | seg | ✅ | ✅ | [yolo-master-seg-n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo-master-seg-n.onnx) | [yolo-master-seg-n_b1_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo-master-seg-n_b1_fp16.engine)<br>[yolo-master-seg-n_b32_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo-master-seg-n_b32_fp16.engine) |
| [FastSAM](./docs) | FastSAM-s.pt | seg | ✅ | ✅ | [FastSAM-s.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/FastSAM-s.onnx) | [FastSAM-s_b1_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/FastSAM-s_b1_fp16.engine)<br>[FastSAM-s_b32_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/FastSAM-s_b32_fp16.engine) |
| [swin-Transformer](./docs) | swin_tiny_patch4_window7_224.pt | cls | ✅ | ✅ | [swin_tiny_patch4_window7_224.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/swin_tiny_patch4_window7_224.onnx) | [swin_tiny_patch4_window7_224_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/swin_tiny_patch4_window7_224_b1_int8.engine)<br>[swin_tiny_patch4_window7_224_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/swin_tiny_patch4_window7_224_b32_int8.engine) |
| [YOLOv13](./docs) | yolov13n.pt | det | ✅ | ✅ | [yolov13n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov13n.onnx) | [yolov13n-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov13n-det_b1_int8.engine)<br>[yolov13n-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov13n-det_b32_int8.engine) |
| [YOLOv12](./docs) | yolov12n.pt | det | ✅ | ✅ | [yolov12n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov12n.onnx) | [yolov12n-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov12n-det_b1_int8.engine)<br>[yolov12n-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov12n-det_b32_int8.engine) |
|  | yolov12n-cls.pt | cls | ✅ | ✅ | [yolov12n-cls.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov12n-cls.onnx) | [yolov12n-cls_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov12n-cls_b1_int8.engine)<br>[yolov12n-cls_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov12n-cls_b32_int8.engine) |
|  | yolov12n-seg.pt | seg | ✅ | ✅ | [yolov12n-seg.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov12n-seg.onnx) | [yolo26n-pose_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov12n-seg_b1_fp16.engine)<br>[yolo26n-pose_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov12n-seg_b32_fp16.engine) |
| [YOLO11](./docs) | yolo11n.pt | det | ✅ | ✅ | [yolo11n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo11n.onnx) | [yolo11n-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-det_b1_int8.engine)<br>[yolo11n-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-det_b32_int8.engine) |
|  | yolo11n-cls.pt | cls | ✅ | ✅ | [yolo11n-cls.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo11n-cls.onnx) | [yolo11n-cls_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-cls_b1_int8.engine)<br>[yolo11n-cls_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-cls_b32_int8.engine) |
|  | yolo11n-seg.pt | seg | ✅ | ✅ | [yolo11n-seg.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo11n-seg.onnx) | [yolo11n-seg_b1_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-seg_b1_fp16.engine)<br>[yolo11n-seg_b32_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-seg_b32_fp16.engine) |
|  | yolo11n-pose.pt | pose | ✅ | ✅ | [yolo11n-pose.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo11n-pose.onnx) | [yolo11n-pose_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-pose_b1_int8.engine)<br>[yolo11n-pose_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo11n-pose_b32_int8.engine) |
| [YOLO-Nas](./docs) | yolo_nas_s.pt | det | ✅ | ✅ | [yolo_nas_s.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolo_nas_s.onnx) | [yolo_nas_s_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo_nas_s_b1_int8.engine)<br>[yolo_nas_s_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolo_nas_s_b32_int8.engine) |
| [YOLOv10](./docs) | yolov10n.pt | det | ✅ | ✅ | [yolov10n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov10n.onnx) | [yolov10n-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov10n-det_b1_int8.engine)<br>[yolov10n-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov10n-det_b32_int8.engine) |
| [YOLOv9](./docs) | yolov9n.pt | det | ✅ | ✅ | [yolov9n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov9n.onnx) | [yolov9t-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov9t-det_b1_int8.engine)<br>[yolov9t-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov9t-det_b32_int8.engine) |
|  | yolov9n-seg.pt | seg | ✅ | ✅ | [yolov9n-seg.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov9n-seg.onnx) | [yolov9c-seg_b1_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov9c-seg_b1_fp16.engine)<br>[yolov9c-seg_b32_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov9c-seg_b32_fp16.engine) |
| [YOLOv8](./docs) | yolov8n.pt | det | ✅ | ✅ | [yolov8n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov8n.onnx) | [yolov8n-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-det_b1_int8.engine)<br>[yolov8n-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-det_b32_int8.engine) |
|  | yolov8n-cls.pt | cls | ✅ | ✅ | [yolov8n-cls.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov8n-cls.onnx) | [yolov8n-cls_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-cls_b1_int8.engine)<br>[yolov8n-cls_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-cls_b1_int8.engine) |
|  | yolov8n-seg.pt | seg | ✅ | ✅ | [yolov8n-seg.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov8n-seg.onnx) | [yolov8n-seg_b1_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-seg_b1_fp16.engine)<br>[yolov8n-seg_b32_fp16.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-seg_b32_fp16.engine) |
|  | yolov8n-pose.pt | pose | ✅ | ✅ | [yolov8n-pose.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov8n-pose.onnx) | [yolov8n-pose_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-pose_b1_int8.engine)<br>[yolov8n-pose_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-pose_b32_int8.engine) |
| [YOLOv8-Face](./docs) | yolov8n-face.pt | face | ✅ | ✅ | [yolov8n-face.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov8n-face.onnx) | [yolov8n-face_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-face_b1_int8.engine)<br>[yolov8n-face_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov8n-face_b32_int8.engine) |
| [YOLOv7](./docs) | yolov7-tiny.pt | det | ✅ | ✅ | [yolov7t.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov7t.onnx) | [yolov7t-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov7t-det_b1_int8.engine)<br>[yolov7t-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov7t-det_b32_int8.engine) |
| [YOLOv6](./docs) | yolov6n.pt | det | ✅ | ✅ | [yolov6n.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov6n.onnx) | [yolov6n-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov6n-det_b1_int8.engine)<br>[yolov6n-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov6n-det_b32_int8.engine) |
| [YOLOv5](./docs) | yolov5s.pt | det | ✅ | ✅ | [yolov5s.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov5s.onnx) | [yolov5s-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov5s-det_b1_int8.engine)<br>[yolov5s-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov5s-det_b32_int8.engine) |
| [YOLOv5-Face](./docs) | yolov5s-face.pt | face | ✅ | ✅ | [yolov5s-face.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov5s-face.onnx) | [yolov5s-face_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov5s-face_b1_int8.engine)<br>[yolov5s-face_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov5s-face_b32_int8.engine) |
| [YOLOv5u](./docs) | yolov5nu.pt | det | ✅ | ✅ | [yolov5nu.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/yolov5nu.onnx) | [yolov5nu-det_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov5nu-det_b1_int8.engine)<br>[yolov5nu-det_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/yolov5nu-det_b32_int8.engine) |
| [YOLOv5-Lite](./docs) | v5lite-g.pt | det | ✅ | ✅ | [v5lite-g.onnx](https://github.com/ppogg/DB-assets/releases/download/v0.01/v5lite-g.onnx) | [v5lite-g_b1_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/v5lite-g_b1_int8.engine)<br>[v5lite-g_b32_int8.engine](https://github.com/ppogg/DB-assets/releases/download/v0.0.2/v5lite-g_b32_int8.engine) |

## Run

| Module | Result |
|:---|:---|
|  ![Face](https://github.com/user-attachments/assets/27bb8887-0d6e-41c0-9c1f-883b3bdf430f)  | ![Detection](https://github.com/user-attachments/assets/e815d5ff-aa25-4bab-a47f-cde270cd6927) |
| ![Segment](https://github.com/user-attachments/assets/20c31c02-02ae-4b44-b432-044f8d8fff93) | ![Action](https://github.com/user-attachments/assets/47272f7c-d6f0-4749-8594-6c19c87483c4) |


## Reference

https://github.com/NVIDIA-AI-IOT/yolo_deepstream

https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation

https://github.com/marcoslucianops/DeepStream-Yolo-Pose

https://github.com/marcoslucianops/DeepStream-Yolo-Seg

## Acknowledgments


