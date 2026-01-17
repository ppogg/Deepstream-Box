import os
import onnx
import torch
import argparse
from onnxsim import simplify
    

def det_forward(self, x):
    z = [] 
    for i in range(self.nl):
        x[i] = self.stems[i](x[i])

        reg_feat = self.reg_convs[i](x[i])
        reg_output = self.reg_preds[i](reg_feat)

        cls_feat = self.cls_convs[i](x[i])
        cls_output = self.cls_preds[i](cls_feat).sigmoid()

        cls_sum = cls_output.sum(1, keepdim=True)
        result = torch.cat((cls_sum, reg_output, cls_output), 1)
        z.append(result)
    return z

def export_onnx(model, file, size=640, dynamic_batch=False, simplify_model=True):

    device = next(model.parameters()).device
    model.float()

    dummy = torch.zeros(1, 3, size, size, device=device)
    model.detect.forward = det_forward.__get__(model.detect, type(model.detect))
    temp_file = file.replace('.onnx', '_temp.onnx') if '.onnx' in file else file + '_temp.onnx'
    
    torch.onnx.export(model, dummy, temp_file, opset_version=17,
                input_names=["images"], output_names=["output0", "output1", "output2"],
                dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}, "output1": 
                                {0: "batch"}, "output2": {0: "batch"}} if dynamic_batch else None)
    
    if simplify_model:
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
                
    elif simplify_model:
        print("onnxsim not available, using original ONNX model")
        if os.path.exists(temp_file):
            os.rename(temp_file, file)
    else:
        if os.path.exists(temp_file):
            os.rename(temp_file, file)

def export(weight, save, size, dynamic, simplify):
    
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")
    
    export_onnx(torch.load(weight, map_location="cpu")["model"], save, size, dynamic_batch=dynamic, simplify_model=simplify)
    print(f"Save onnx to {save}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='export yolov6')
    parser.add_argument("--weight", type=str, default="weights/yolov6n.pt", help="export pt file")
    parser.add_argument("--save", type=str, required=False, help="export onnx file")
    parser.add_argument("--size", type=int, default=640, help="export input size")
    parser.add_argument("--dynamic", action="store_true", help="export dynamic batch")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify", default=True, help="disable onnx simplification")

    args = parser.parse_args()
    export(args.weight, args.save, args.size, args.dynamic, args.simplify)