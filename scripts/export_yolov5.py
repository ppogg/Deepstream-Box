import sys
import os

# Add the current directory to PYTHONPATH for YoloV5
sys.path.insert(0, os.path.abspath("."))
pydir = os.path.dirname(__file__)

import warnings
import argparse

# PyTorch
import torch
import torch.nn as nn

# YoloV7
from models.yolo import Model
from models.common import Conv

# ONNX Simplifier
try:
    import onnx
    import onnxsim
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("Warning: onnxsim not installed. Install with: pip install onnxsim")

# Disable all warning
warnings.filterwarnings("ignore")

def load_yolov7_model(weight, device) -> Model:
    model = torch.load(weight, map_location=device)["model"]
    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model
    

def export_onnx_v1(model : Model, file, size=640, dynamic_batch=False, simplify_model=True):

    device = next(model.parameters()).device
    model.float()

    dummy = torch.zeros(1, 3, size, size, device=device)
    model.model[-1].concat = True
    grid_old_func = model.model[-1]._make_grid
    model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())
    temp_file = file.replace('.onnx', '_temp.onnx') if '.onnx' in file else file + '_temp.onnx'
    
    torch.onnx.export(model, dummy, temp_file, opset_version=17, 
        input_names=["images"], output_names=["outputs"], 
        dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None
    )
    
    if simplify_model and ONNXSIM_AVAILABLE:
        try:
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(temp_file)
            onnx.checker.check_model(onnx_model)
            input_shape = {'images': [1, 3, size, size]} if not dynamic_batch else None
            model_simp, check = simplify(onnx_model, 
                                        test_input_shapes=input_shape,
                                        dynamic_input_shape=dynamic_batch,
                                        skip_shape_inference=False)
            
            if check:
                onnx.save(model_simp, file)
                print(f"Simplification successful!")
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                print("Simplification check failed, using original model")
                os.rename(temp_file, file)
                
        except Exception as e:
            print(f"Simplification failed: {e}")
            print("Using original ONNX model")
            if os.path.exists(temp_file):
                os.rename(temp_file, file)

    elif simplify_model and not ONNXSIM_AVAILABLE:
        print("onnxsim not available, using original ONNX model")
        if os.path.exists(temp_file):
            os.rename(temp_file, file)
    else:
        if os.path.exists(temp_file):
            os.rename(temp_file, file)


def new_forward(self, x):
    z = [] 
    for i in range(self.nl):
        x[i] = self.m[i](x[i])
        print(x[i].shape)
        z.append(x[i])
    return z


def export_onnx_v2(model : Model, file, size=640, dynamic_batch=False, simplify_model=True):

    device = next(model.parameters()).device
    model.float()

    dummy = torch.zeros(1, 3, size, size, device=device)
    model.model[-1].forward = new_forward.__get__(model.model[-1], type(model.model[-1]))
    temp_file = file.replace('.onnx', '_temp.onnx') if '.onnx' in file else file + '_temp.onnx'
    
    torch.onnx.export(model, dummy, temp_file, opset_version=17, 
        input_names=["images"], output_names=["output0", "output1", "output2"], 
        dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None
    )
    
    if simplify_model and ONNXSIM_AVAILABLE:
        try:
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(temp_file)
            onnx.checker.check_model(onnx_model)
            input_shape = {'images': [1, 3, size, size]} if not dynamic_batch else None
            model_simp, check = simplify(onnx_model, 
                                        test_input_shapes=input_shape,
                                        dynamic_input_shape=dynamic_batch,
                                        skip_shape_inference=False)
            
            if check:
                onnx.save(model_simp, file)
                print(f"Simplification successful!")
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                print("Simplification check failed, using original model")
                os.rename(temp_file, file)
                
        except Exception as e:
            print(f"Simplification failed: {e}")
            print("Using original ONNX model")
            if os.path.exists(temp_file):
                os.rename(temp_file, file)
                
    elif simplify_model and not ONNXSIM_AVAILABLE:
        print("onnxsim not available, using original ONNX model")
        if os.path.exists(temp_file):
            os.rename(temp_file, file)
    else:
        if os.path.exists(temp_file):
            os.rename(temp_file, file)

def export(weight, save, size, dynamic, grid, simplify):
    
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")
    
    if not ONNXSIM_AVAILABLE and simplify:
        print("Warning: onnxsim not installed. Install with: pip install onnxsim")
        print("Proceeding without simplification...")
    
    if grid == True:
        export_onnx_v1(torch.load(weight, map_location="cpu")["model"], save, size, dynamic_batch=dynamic, simplify_model=simplify)
    else:
        export_onnx_v2(torch.load(weight, map_location="cpu")["model"], save, size, dynamic_batch=dynamic, simplify_model=simplify)
    print(f"Save onnx to {save}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='export yolov5')
    parser.add_argument("--weight", type=str, default="weights/yolov5s.pt", help="export pt file")
    parser.add_argument("--save", type=str, required=False, help="export onnx file")
    parser.add_argument("--size", type=int, default=640, help="export input size")
    parser.add_argument("--dynamic", type=bool, default=True, help="export dynamic batch")
    parser.add_argument('--grid', type=bool, default=False, help='export Detect() layer grid')
    parser.add_argument("--no-simplify", action="store_false", dest="simplify", default=True, help="disable onnx simplification")

    args = parser.parse_args()
    export(args.weight, args.save, args.size, args.dynamic, args.grid, args.simplify)