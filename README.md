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

|Model | type | task | tutorial | quantize | dynamic batch
|:-:|:-:|:-:|:-:|:-:|:-:|
|[YOLO26](./docs) | n, s, m, l, x | cls, det, pose, seg | ✅| ✅| ✅
|[YOLO-Master](./docs) | n, s, m, l, x | cls, det, seg | ✅| ✅| ✅
|[Fast-SAM](./docs) | s, x | seg | ✅| ❌| ❌
|[Swim-transformer](./docs) | tiny, base, large | cls | ✅| ✅| ✅
|[RT-DETR](./docs) | l, x | det | ❌| ✅| ✅
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
<table border="0">
  <!-- 第一行：Detection -->
  <tr>
    <td align="center" style="font-weight: bold; background-color: #f6f8fa; vertical-align: middle; padding: 10px; width: 25%">
      <b>Detection</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/e815d5ff-aa25-4bab-a47f-cde270cd6927" alt="Detection Result" style="width:100%; display:block; border-radius:4px;">
    </td>
  </tr>
  <!-- 第二行：Action -->
  <tr>
    <td align="center" style="font-weight: bold; background-color: #f6f8fa; vertical-align: middle; padding: 10px;width: 20%">
      <b>Action</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/47272f7c-d6f0-4749-8594-6c19c87483c4" alt="Action Result" style="width:100%; display:block; border-radius:4px;">
    </td>
  </tr>
  <!-- 第三行：Segment -->
  <tr>
    <td align="center" style="font-weight: bold; background-color: #f6f8fa; vertical-align: middle; padding: 10px;width: 20%">
      <b>Segment</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/20c31c02-02ae-4b44-b432-044f8d8fff93" alt="Segmentation Result" style="width:100%; display:block; border-radius:4px;">
    </td>
  </tr>
  <!-- 第四行：Face -->
  <tr>
    <td align="center" style="font-weight: bold; background-color: #f6f8fa; vertical-align: middle; padding: 10px;width: 20%">
      <b>Face</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/1b0663d6-db3b-4dfe-9d18-2963740e3f25" alt="Face Result" style="width:100%; display:block; border-radius:4px;">
    </td>
  </tr>
</table>

## Reference

https://github.com/NVIDIA-AI-IOT/yolo_deepstream

https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation

https://github.com/marcoslucianops/DeepStream-Yolo-Pose

https://github.com/marcoslucianops/DeepStream-Yolo-Seg

## Acknowledgments


