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

1. Ubantu = 22.04 (PC or Jeston)
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

|Model | type | task | tutorial | quantize | dynamic batch
|:-:|:-:|:-:|:-:|:-:|:-:|
|[YOLO26](./docs) | n, s, m, l, x | cls, det, pose, seg | ✅| ✅| ✅
|[YOLO-Master](./docs) | n, s, m, l, x | cls, det, seg | ✅| ✅| ✅
|[Fast-SAM](./docs) | s, x | seg | ✅| ❌| ❌
|[Swim-transformer](./docs) | tiny, base, large | cls | ✅| ✅| ✅
|[YOLO13](./docs) | n, s, m, l, x | det | ✅| ❌| ✅
|[YOLO12](./docs) | n, s, m, l, x | cls, det, seg | ✅| ✅| ✅
|[YOLO11](./docs) | n, s, m, l, x | cls, det, pose, seg | ✅| ✅| ✅
|[YOLO10](./docs) | n, s, m, l, x | det | ✅| ✅| ✅
|[YOLOv9](./docs) | t, s, m, c, e | det, seg | ✅| ✅| ✅
|[YOLOv8](./docs) | n, s, m, l, x | cls, det, pose, seg | ✅| ✅| ✅
|[YOLOv7](./docs) | T, X, W6, E6, D6, E6E | det | ✅| ✅| ✅
|[YOLOv6](./docs) | N, S, M, L | det | ✅| ✅| ✅
|[YOLOv5-Lite](./docs) | e, s, c, g | det | ✅| ✅| ✅
|[YOLOv5u](./docs) | n, s, m, l, x | det | ✅| ✅| ✅
|[YOLOv5](./docs) | n, s, m, l, x |  det| ✅| ✅| ✅

## Run
### Detection
<video src="[https://github.com/你的用户名/仓库名/raw/main/你的视频文件名.mp4](https://release-assets.githubusercontent.com/github-production-release-asset/1136074999/0b16768f-9987-42d5-9f3a-f6d77d8d92b5?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-01-17T03%3A56%3A55Z&rscd=attachment%3B+filename%3Ddet_result.mp4&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-01-17T02%3A56%3A48Z&ske=2026-01-17T03%3A56%3A55Z&sks=b&skv=2018-11-09&sig=VYWOx3vt2it9Ujg0t74OGYQvLUDmsBOIzyxrDNGYkZ4%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc2ODYyMDA1OCwibmJmIjoxNzY4NjE5NzU4LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.e5aupBwJpkn2E5WrVkojQaciQtl5p8SLx5i-fBrsE1E&response-content-disposition=attachment%3B%20filename%3Ddet_result.mp4&response-content-type=application%2Foctet-stream)" width="800" controls></video>
### Action
<video src='https://github.com/ppogg/DB-assets/releases/download/v0.0/pose_result.mp4' controls width='100%'></video>
### Segment
<video src='https://github.com/ppogg/DB-assets/releases/download/v0.0/seg_result.mp4' controls width='100%'></video>

## Reference

https://github.com/NVIDIA-AI-IOT/yolo_deepstream

https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation

https://github.com/marcoslucianops/DeepStream-Yolo-Pose

https://github.com/marcoslucianops/DeepStream-Yolo-Seg

## Acknowledgments


