# YOLO-Master usage

**NOTE**: Please follow the steps to operate.
* [Preparatory Work](#preparatory-work)
* [Export Onnx](#export-onnx)
* [Build Engine](#build-engine)
* [Compile Target](#compile-target)
* [Edit the config file](#edit-the-config-file)
* [Running the program](#testing-the-model)
##


### 1. Preparatory Work

#### 1.1 Download the YOLO-Master repo and install the requirements

```
git clone https://github.com/isLinXu/YOLO-Master.git
```
#### 1.2 Download the deepstreamx repo

```
git clone https://github.com/ppogg/Deepstream-Box.git
```

#### 1.3 Copy export file

Copy `export_ultralytics.py`.

```
cp deepstreamx/scripts/export_ultralytics.py YOLO-Master/ultralytics/
```

#### 1.4 Download the model

Download the `.pt` from [YOLO Master](https://github.com/isLinXu/YOLO-Master/releases) releases, like

**a. Detection task** 👇
```
wget https://github.com/isLinXu/YOLO-Master/releases/download/v0.0.0/yolo-master-esmoe-n.pt
```
b. Classification task 👇
```
wget https://github.com/isLinXu/YOLO-Master/releases/download/v0.0.0/yolo-master-cls-n.pt
```
c. Segment task 👇
```
wget https://github.com/isLinXu/YOLO-Master/releases/download/v0.0.0/yolo-master-seg-n.pt
```
##

### 2. Export Onnx

#### 2.1 Modify the onnx parameters
| Argument | 	Type | 	Default | 	Description |
| :---: | :---: | :---: | :---: |
| **weight** | str | yolo-master-esmoe-n.pt |The original .pt model. |
| **size** | int |640 |Network input size. |
| **task** | int |0 |Task type (0-det/1-kpt/2-seg/3-nas/4-cls). |
| **dynamic** | bool |False |Dynamic processing. |

#### 2.2 Generate the ONNX model file.
**a. export det onnx.**
```
python export_ultralytics.py --weight yolo-master-esmoe-n.pt --task 0 --dynamic
```
b. export cls onnx.

```
python export_ultralytics.py --weight yolo-master-cls-n.pt --task 4 --dynamic
```
c. export seg onnx.

```
python export_ultralytics.py --weight yolo-master-seg-n.pt --task 2
```
**NOTE**: Segment task only suppurt `batch=1`.
##

### 3. Build Engine

#### 3.1 Modify the tensort parameters
| Argument | 	Type | 	Default | 	Description |
|:---: |:---: |:---: |:---: |
| **onnx** | str | - |The onnx model path. |
| **calib_data** | str |- |Calibration data directory path (required for INT8 mode). |
| **precision** | str |int8 |Quantization precision: int8, fp16, tf32, or fp32 (default: int8). |
| **batch_size** | int |32 |Batch size (default: 32). |
| **num_calib** | int |320 |Number of calibration images (default: 100, INT8 only). |
| **check_height** | int |640 |Fixed input height (if ONNX has dynamic shape) (default: 640). |
| **check_width** | int |640 |Fixed input width (if ONNX has dynamic shape) (default: 640). |
| **workspace** | int |2048 |TensorRT workspace size(MB) (default: 2048). |
| **layer_precision** | str |best |Layer precision strategy: best, high, fast (default: best). |

#### 3.2 Build the Engine model file.
**a. build det int8 engine.**

```
python ptq.py --onnx yolo-master-esmoe-n.onnx --calib_data images/ --batch_size 32
```
b. export cls int8 engine.

```
python ptq.py --onnx yolo-master-cls-n.onnx --calib_data images/ --batch_size 32
```
c. export seg int8 engine.

```
python ptq.py --onnx yolo-master-seg-n.onnx
```
**NOTE**: Segment task only suppurt `batch=1` and `fp16`.
##

### 4. Compile Target

#### 4.1 Switch to the following directory.

```
cd deepstreamx/nvdsinfer_custom_layers
```

#### 4.2 Set compilation information in the Makefile.

```
CUDA_VER = 12.2
DS_ROOT = path_to_deepstream_sdk
```

**NOTE**: Recommended version as follows:
· CUDA version >= 12.2
· deepstream version >= 7.0

#### 4.3 compile target file

```
make -j16
```
After compilation, `deepstream-pose` and `nvdsinfer_custom_layers.so` will be generated in the `target` folder.
Note. `deepstream-pose` is used for pose tasks. `nvdsinfer_custom_layers.so` is the link library for all tasks.
##

### 5. Edit the config file

#### 5.1 Detection task

Please adjust `deepstream_app_config_det.txt`.

```
...
[primary-gie]
...
config-file=master/config_infer_dbox_master.txt
```
Then adjust `config_infer_dbox_master.txt`.
```
[property]
...
model-engine-file=../../model_zoo/master/yolo-master-esmoe-n_int8.engine
network-mode=1
network-type=0
cluster-mode=2
num-detected-classes=80
parse-bbox-func-name=NvDsInferParseCustomMaster_cuda
```

#### 5.2 Segment task

Please adjust `deepstream_app_config_seg.txt`.

```
...
[primary-gie]
...
config-file=master/config_infer_mask_master.txt
```
Then adjust `config_infer_mask_master.txt`.
```
[property]
...
model-engine-file=../../model_zoo/master/yolo-master-seg-n_fp16.engine
network-mode=2
network-type=3
cluster-mode=4
parse-bbox-func-name=NvDsInferParseMasterSeg
output-instance-mask=1
segmentation-threshold=0.5
```

**NOTE 1**: **YOLO Series** must use letterbox to preprocess.

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```
**NOTE 2**: **Seg Task** only support fp16 mode.
**NOTE 3**: **Pose Task** and **Cls Task** Similar to **Det Task**.

##

### 6. Running the program

**Testing the program**

#### 6.1 For detection task.
```
deepstream-app -c deepstream_app_config_det.txt
```
#### 6.2 For classifier task.

```
deepstream-app -c deepstream_app_config_cls.txt
```
#### 6.3 For seg task.
```
deepstream-app -c deepstream_app_config_seg.txt
```
**NOTE**: If you encounter any problems, please raise an `issue`. I will reply as soon as possible.

