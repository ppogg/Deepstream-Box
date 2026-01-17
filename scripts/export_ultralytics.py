import sys
import os
import argparse
import numpy as np

# Add the current directory to PYTHONPATH for YoloV8
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/pogg/Project/ultralytics-8.4.0")

import warnings
import torch

from ultralytics import YOLO
from ultralytics import NAS


try:
    import onnx
    from onnx import helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx not installed. Install with: pip install onnx")

try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("Warning: onnxsim not installed. Install with: pip install onnxsim")

warnings.filterwarnings("ignore")

def cls_forward(self, x):
    x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
    y = x.softmax(1) 
    scores, id = torch.max(y, 1)
    return torch.cat([id.unsqueeze(1), scores.unsqueeze(1)], dim=1)

def det_forward(self, x):
    z = [] 
    for i in range(self.nl):
        reg = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        cls_sum = cls.sum(1, keepdim=True)
        result = torch.cat((cls_sum, reg, cls), 1)
        z.append(result)
    return z

def det26_forward(self, x):
    z = [] 
    for i in range(self.nl):
        reg = self.one2one_cv2[i](x[i])
        cls = self.one2one_cv3[i](x[i]).sigmoid()
        cls_sum = cls.sum(1, keepdim=True)
        result = torch.cat((cls_sum, reg, cls), 1)
        z.append(result)
    return z

def kpt_forward(self, x):
    z = [] 
    for i in range(self.nl):
        cls = self.cv3[i](x[i]).sigmoid()
        kpt = self.cv4[i](x[i])
        result = torch.cat((cls, kpt), 1)
        z.append(result)
    return z

def kpt26_forward(self, x):
    z = [] 
    for i in range(self.nl):
        cls = self.one2one_cv3[i](x[i]).sigmoid()
        one2one_cv4 = self.one2one_cv4[i](x[i])
        kpt = self.one2one_cv4_kpts[i](one2one_cv4)
        result = torch.cat((cls, kpt), 1)
        z.append(result)
    return z

def nas_forward(self, x):
    z = [] 
    for i, head_name in enumerate(['head1', 'head2', 'head3']):
        head = getattr(self, head_name)
        x_i = x[i]
        stem_out = head.stem(x_i)

        cls_feat = head.cls_convs(stem_out)
        cls_out = head.cls_pred(cls_feat).sigmoid()
        cls_sum = cls_out.sum(1, keepdim=True)

        reg_feat = head.reg_convs(stem_out)
        reg_out = head.reg_pred(reg_feat)
        
        result = torch.cat((cls_sum, reg_out, cls_out), 1)
        z.append(result)
    
    return z


def export_onnx(weight_file, file, size=640, opsetsion=17, task=0, dynamic_batch=True, simplify_model=True):
    
    temp_file = file.replace('.onnx', '_temp.onnx') if '.onnx' in file else file + '_temp.onnx'

    if task == 2:
        print("Exporting end-to-end segmentation model with custom post-processing...")
        model = YOLO(weight_file)
        temp_file = model.export(format='onnx', imgsz=size, opset=opsetsion, simplify=True, nms=True, batch=1)

        model = onnx.load(temp_file)
        graph = model.graph
        
        output0_name = graph.output[0].name
        output1_name = graph.output[1].name

        MAX_DETS = graph.output[0].type.tensor_type.shape.dim[1].dim_value
        MC = graph.output[0].type.tensor_type.shape.dim[2].dim_value - 6

        slice_bbox_node = helper.make_node(
        'Slice',
        inputs=[output0_name, 'starts_bbox', 'ends_bbox', 'axes_bbox'],
        outputs=['bbox_raw']
        )
        starts_bbox = helper.make_tensor('starts_bbox', TensorProto.INT32, [1], [0])
        ends_bbox = helper.make_tensor('ends_bbox', TensorProto.INT32, [1], [4])
        axes_bbox = helper.make_tensor('axes_bbox', TensorProto.INT32, [1], [2])

        slice_scores_node = helper.make_node(
            'Slice',
            inputs=[output0_name, 'starts_scores', 'ends_scores', 'axes_scores'],
            outputs=['scores_raw']
        )
        starts_scores = helper.make_tensor('starts_scores', TensorProto.INT32, [1], [4])
        ends_scores = helper.make_tensor('ends_scores', TensorProto.INT32, [1], [6])
        axes_scores = helper.make_tensor('axes_scores', TensorProto.INT32, [1], [2])

        slice_mask_coeff_node = helper.make_node(
            'Slice',
            inputs=[output0_name, 'starts_mask', 'ends_mask', 'axes_mask'],
            outputs=['mask_coeff_raw']
        )
        starts_mask = helper.make_tensor('starts_mask', TensorProto.INT32, [1], [6])
        ends_mask = helper.make_tensor('ends_mask', TensorProto.INT32, [1], [38])
        axes_mask = helper.make_tensor('axes_mask', TensorProto.INT32, [1], [2])

        axes_0 = helper.make_tensor('axes_0', TensorProto.INT64, [1], [0])
        axes_1 = helper.make_tensor('axes_1', TensorProto.INT64, [1], [1])
        axes_1_const = helper.make_tensor('axes_1_const', TensorProto.INT64, [1], [1])

        squeeze_bbox_node = helper.make_node(
            'Squeeze',
            inputs=['bbox_raw', 'axes_0'],
            outputs=['bbox_squeezed']
        )

        squeeze_scores_node = helper.make_node(
            'Squeeze',
            inputs=['scores_raw', 'axes_0'],
            outputs=['scores_squeezed']
        )
        
        squeeze_coeffs_node = helper.make_node(
            'Squeeze',
            inputs=['mask_coeff_raw', 'axes_0'],
            outputs=['coeffs_squeezed']
        )

        batch_idx_val = np.zeros((MAX_DETS,), dtype=np.int64)
        batch_idx_tensor = helper.make_tensor('batch_idx_const', TensorProto.INT64 , [MAX_DETS], batch_idx_val.flatten())

        roi_align_node = helper.make_node(
            'RoiAlign',
            inputs=[output1_name, 'bbox_squeezed', 'batch_idx_const'],
            outputs=['pooled_proto'],
            output_height=80,
            output_width=80,
            sampling_ratio=2,
            spatial_scale=0.25,
            mode='avg',
            coordinate_transformation_mode='half_pixel'
        )

        shape_pooled_val = np.array([MAX_DETS, MC, 80*80], dtype=np.int64)
        shape_pooled_tensor = helper.make_tensor('shape_pooled', TensorProto.INT32, [3], shape_pooled_val)
        reshape_pooled_node = helper.make_node(
            'Reshape',
            inputs=['pooled_proto', 'shape_pooled'],
            outputs=['pooled_reshaped']
        )

        shape_coeffs_val = np.array([MAX_DETS, 1, MC], dtype=np.int64) 
        shape_coeffs_tensor = helper.make_tensor('shape_coeffs', TensorProto.INT32, [3], shape_coeffs_val)
        reshape_coeffs_node = helper.make_node(
            'Reshape',
            inputs=['coeffs_squeezed', 'shape_coeffs'],
            outputs=['coeffs_reshaped']
        )
    
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['coeffs_reshaped', 'pooled_reshaped'],
            outputs=['matmul_out']
        )
    
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=['matmul_out'],
            outputs=['masks_sigmoid'] # [100, 1, 6400]
        )

        squeeze_masks_node = helper.make_node(
            'Squeeze',
            inputs=['masks_sigmoid', 'axes_1'], # axes_1 = [1]
            outputs=['masks_final']
        )

        concat_final_node = helper.make_node(
            'Concat',
            inputs=['bbox_squeezed', 'scores_squeezed', 'masks_final'],
            outputs=['final_output_raw'],
            axis=1
        )

        unsqueeze_final_node = helper.make_node(
            'Unsqueeze',
            inputs=['final_output_raw', 'axes_0'],
            outputs=['final_output']
        )

        new_nodes = [
            slice_bbox_node, slice_scores_node, slice_mask_coeff_node,
            squeeze_bbox_node, squeeze_scores_node, squeeze_coeffs_node,
            roi_align_node, reshape_pooled_node, reshape_coeffs_node,
            matmul_node, sigmoid_node, squeeze_masks_node, 
            concat_final_node, unsqueeze_final_node
        ]
        
        new_initializers = [
            starts_bbox, ends_bbox, axes_bbox,
            starts_scores, ends_scores, axes_scores,
            starts_mask, ends_mask, axes_mask,
            axes_0, axes_1, axes_1_const,
            batch_idx_tensor, 
            shape_pooled_tensor, shape_coeffs_tensor
        ]
        
        graph.node.extend(new_nodes)
        graph.initializer.extend(new_initializers)
        
        graph.ClearField('output')
        final_output = helper.make_tensor_value_info('final_output', TensorProto.FLOAT, [1, MAX_DETS, 6406])
        graph.output.extend([final_output])
        
        onnx.save(model, temp_file)
                
    elif task == 3:

        model = NAS(weight_file)  
        model = model.model
        model.to(device="cpu")
        model.float()
        model.eval()

        device = next(model.parameters()).device
        model.float()
        dummy = torch.zeros(1, 3, size, size, device=device)

        if hasattr(model, 'model'):
            tmp_model = model.model
        else:
            tmp_model = model
            

        model.heads.forward = nas_forward.__get__(model.heads, type(model.heads))

    else:
        model = YOLO(weight_file)
        model = model.model
        model.to(device="cpu")
        model.float()
        model.eval()

        device = next(model.parameters()).device
        model.float()
        dummy = torch.zeros(1, 3, size, size, device=device)
        
        if hasattr(model, 'model'):
            tmp_model = model.model
        else:
            tmp_model = model

        if task == 0: tmp_model[-1].forward = det_forward.__get__(tmp_model[-1], type(tmp_model[-1]))
        elif task == 1: tmp_model[-1].forward = kpt_forward.__get__(tmp_model[-1], type(tmp_model[-1]))
        elif task == 4: tmp_model[-1].forward = cls_forward.__get__(tmp_model[-1], type(tmp_model[-1]))
        elif task == 5: tmp_model[-1].forward = det26_forward.__get__(tmp_model[-1], type(tmp_model[-1]))
        elif task == 6: tmp_model[-1].forward = kpt26_forward.__get__(tmp_model[-1], type(tmp_model[-1]))

    if task == 0 or task == 1 or task == 3 or task == 5 or task == 6:
        torch.onnx.export(model, dummy, temp_file, opset_version=opsetsion,
                        input_names=["images"], output_names=["output0", "output1", "output2"],
                        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}, "output1": 
                                      {0: "batch"}, "output2": {0: "batch"}} if dynamic_batch else None)
        
    elif task == 4:
         torch.onnx.export(model, dummy, temp_file, opset_version=opsetsion,
                        input_names=["images"], output_names=["outputs"],
                        dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} 
                        if dynamic_batch else None)
        
    if simplify_model:
        print("Simplifying ONNX model...")
        final_file, check = simplify(onnx.load(temp_file))
        onnx.save(final_file, file)
        if os.path.exists(temp_file) and task != 2: os.remove(temp_file)
        print(f"Simplification successful! Final model saved to {file}")


def export(weight, size, opset, task, dynamic, simplify):
    name = os.path.basename(weight).rsplit('.', 1)[0]
    save = os.path.join(os.path.dirname(weight), name + ".onnx")
    export_onnx(weight, save, size, opset, task, dynamic_batch=dynamic, simplify_model=simplify)
    print(f"Export complete. Model saved to {save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ultralytics export')
    parser.add_argument("--weight", type=str, default="yolo26l.pt", help="export pt file")
    parser.add_argument("--size", type=int, default=640, help="export input size")
    parser.add_argument("--opset", type=int, default=17, help="opset version")
    parser.add_argument("--task", type=int, default=0, help="export task type (0-det/1-kpt/2-seg/3-nas/4-cls/5-yolo26_det/6-yolo26_kpt)")
    parser.add_argument("--dynamic", type=bool, default=False, help="export dynamic batch, only support det/kpt")
    parser.add_argument("--simplify", action="store_true", default=True, help="disable onnx simplification")

    args = parser.parse_args()
    export(args.weight, args.size, args.opset, args.task, args.dynamic, args.simplify)