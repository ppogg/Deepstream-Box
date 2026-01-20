import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
TRT_LOGGER = trt.Logger()

def parse_arguments():
    parser = argparse.ArgumentParser(description='TensorRT Model Quantization Tool')
    parser.add_argument('--onnx', type=str, default="/home/pogg/DeepStream/Deepstream-Box/model_zoo/yolo26/yolo26n-pose.onnx", help='ONNX model path')
    parser.add_argument('--calib_data', type=str, default="/home/pogg/Dataset/coco/images/train2017", help='Calibration data directory path (required for INT8 mode)')
    parser.add_argument('--precision', type=str, default='fp32', choices=['int8', 'fp16', 'tf32', 'fp32'], help='Quantization precision: int8, fp16, tf32, or fp32 (default: int8)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--num_calib', type=int, default=320, help='Number of calibration images (default: 100, INT8 only)')
    parser.add_argument('--output', type=str, default=None, help='Output engine file path, default same directory as ONNX')
    parser.add_argument('--cache_file', type=str, default=None, help='Calibration cache file path, default same directory as ONNX')
    parser.add_argument('--check_height', type=int, default=640, help='Fixed input height (if ONNX has dynamic shape) (default: 640)')
    parser.add_argument('--check_width', type=int, default=640, help='Fixed input width (if ONNX has dynamic shape) (default: 640)')
    parser.add_argument('--workspace', type=int, default=10240, help='TensorRT workspace size(MB) (default: 1024)')
    parser.add_argument('--no_fallback', action='store_true', help='Disable auto fallback (do not try FP16 if INT8 fails)')
    parser.add_argument('--layer_precision', type=str, default='best', choices=['best', 'high', 'fast'], help='Layer precision strategy: best, high, fast (default: best)')
    parser.add_argument('--sparsity', action='store_true', help='Enable sparsity optimization')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    
    return parser.parse_args()

def load_coco_image(cocodir, num_images=None, verbose=False):
    if verbose:
        print(f"\n{'='*60}")
        print("Start loading calibration images...")
    
    if not os.path.exists(cocodir):
        print(f"Error: Directory does not exist - {cocodir}")
        return None
    
    files = os.listdir(cocodir)
    jpg_files = [file for file in files if file.endswith(".jpg")]
    
    if not jpg_files:
        print(f"Error: No jpg files found in directory - {cocodir}")
        return None
    
    if verbose:
        print(f"Found {len(jpg_files)} jpg files")
    
    if num_images is not None:
        np.random.seed(32)
        np.random.shuffle(jpg_files)
        selected_files = jpg_files[:min(num_images, len(jpg_files))]
        if verbose:
            print(f"Randomly selected {len(selected_files)} images")
    else:
        selected_files = jpg_files
        if verbose:
            print(f"Will load all {len(selected_files)} images")
    
    datas = []
    success_count = 0
    
    for i, file in enumerate(selected_files):
        file_path = os.path.join(cocodir, file)
        img = cv2.imread(file_path)
        if img is None:
            continue
        
        from_ = img.shape[1], img.shape[0]
        to_ = 640, 640
        scale = min(to_[0] / from_[0], to_[1] / from_[1])
        
        M = np.array([
            [scale, 0, -scale * from_[0] * 0.5 + to_[0] * 0.5 + scale * 0.5 - 0.5 + 16],
            [0, scale, -scale * from_[1] * 0.5 + to_[1] * 0.5 + scale * 0.5 - 0.5 + 16],
        ])
        
        input_img = cv2.warpAffine(img, M, (640, 640), borderValue=(114, 114, 114))
        input_img = input_img[..., ::-1].transpose(2, 0, 1)[None]
        input_img = (input_img / 255.0).astype(np.float32)
        datas.append(input_img)
        success_count += 1
    
    if not datas:
        print("Error: Failed to load any images!")
        return None
    
    all_data = np.concatenate(datas, axis=0)
    if verbose:
        print(f"\nImage loading completed!")
        print(f"Successfully loaded: {success_count} images")
        print(f"Final data shape: {all_data.shape}")
        print(f"{'='*60}")
    
    return all_data

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data_dir, cache_file, batch_size=16, num_calib_images=100, verbose=False):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        if verbose:
            print(f"\n{'='*60}")
            print("Initialize INT8 Calibrator")
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        
        if os.path.exists(cache_file):
            if verbose:
                print(f"Found existing calibration cache file: {cache_file}")
            self.data = None
            self.device_input = None
        else:
            if verbose:
                print(f"Calibration cache file not found, start loading calibration data...")
            self.data = load_coco_image(training_data_dir, num_calib_images, verbose)
            
            if self.data is None:
                raise ValueError("Failed to load calibration data!")
            
            total_images = self.data.shape[0]
            if total_images < batch_size:
                raise ValueError(f"Insufficient calibration data! Need at least {batch_size} images, but only loaded {total_images}")
            
            self.data = np.ascontiguousarray(self.data)
            
            bytes_per_batch = self.data[0:batch_size].nbytes
            if verbose:
                print(f"Allocate GPU memory: {bytes_per_batch:,} bytes ({bytes_per_batch/(1024*1024):.2f} MB)")
            
            try:
                self.device_input = cuda.mem_alloc(bytes_per_batch)
                if verbose:
                    print("GPU memory allocation successful")
            except Exception as e:
                print(f"GPU memory allocation failed: {e}")
                raise
        
        if verbose:
            print(f"{'='*60}")
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.data is None:
            return None
        
        if self.current_index + self.batch_size > len(self.data):
            return None
        
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        batch_flat = batch.ravel()
        cuda.memcpy_htod(self.device_input, batch_flat)
        return [self.device_input]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_engine(onnx_file, calib, batch_size=16, precision="int8", 
                 check_height=640, check_width=640, workspace_mb=1024, 
                 fallback=True, layer_precision="best", sparsity=False, verbose=False):
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Start building TensorRT engine")
        print(f"ONNX file: {onnx_file}")
        print(f"Batch size: {batch_size}")
        print(f"Target precision: {precision.upper()}")
        print(f"Using calibrator: {'Yes' if calib and precision == 'int8' else 'No'}")
        print(f"Workspace: {workspace_mb} MB")
        print(f"Layer precision strategy: {layer_precision}")
        print(f"Sparsity optimization: {'Enabled' if sparsity else 'Disabled'}")
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network:
        
        if verbose:
            print(f"\nParsing ONNX model: {onnx_file}")
        with trt.OnnxParser(network, TRT_LOGGER) as parser:
            if not parser.parse_from_file(onnx_file):
                print("ONNX parsing failed!")
                for i in range(parser.num_errors):
                    print(f"  Error {i}: {parser.get_error(i)}")
                return None, "ONNX parsing failed"
        
        if verbose:
            print("ONNX model parsed successfully")
        
        if network.num_inputs == 0:
            print("Error: Network has no inputs!")
            return None, "Network has no inputs"
        
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        if verbose:
            print(f"Original input shape: {input_shape}")
        
        precision_chain = []
        
        if precision == "int8":
            if not fallback:
                precision_chain = [("INT8", [trt.BuilderFlag.INT8], True)]
            else:
                precision_chain = [
                    ("INT8", [trt.BuilderFlag.INT8], True),
                    ("FP16", [trt.BuilderFlag.FP16], False),
                    ("TF32", [trt.BuilderFlag.TF32], False),
                    ("FP32", [], False)
                ]
        elif precision == "fp16":
            if not fallback:
                precision_chain = [("FP16", [trt.BuilderFlag.FP16], False)]
            else:
                precision_chain = [
                    ("FP16", [trt.BuilderFlag.FP16], False),
                    ("TF32", [trt.BuilderFlag.TF32], False),
                    ("FP32", [], False)
                ]
        elif precision == "tf32":
            if not fallback:
                precision_chain = [("TF32", [trt.BuilderFlag.TF32], False)]
            else:
                precision_chain = [
                    ("TF32", [trt.BuilderFlag.TF32], False),
                    ("FP32", [], False)
                ]
        elif precision == "fp32":
            precision_chain = [("FP32", [], False)]
        
        last_error = None
        
        for mode_idx, (mode_name, flags, needs_calib) in enumerate(precision_chain):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Attempting to build {mode_name} engine...")
            
            if mode_name == "INT8" and (not calib or needs_calib):
                if not calib:
                    if verbose:
                        print("Skip INT8 mode (no calibrator)")
                    if fallback:
                        print("Will try next precision mode...")
                        continue
                    else:
                        return None, "INT8 mode requires calibrator"
            
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
            
            for flag in flags:
                config.set_flag(flag)
            
            if layer_precision == "high":
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            elif layer_precision == "fast":
                config.set_flag(trt.BuilderFlag.DIRECT_IO)
            
            if sparsity:
                try:
                    if hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
                        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
                        if verbose:
                            print("Sparsity optimization enabled")
                except:
                    if verbose:
                        print("Current TensorRT version does not support sparsity optimization")
            
            if mode_name == "INT8" and needs_calib and calib:
                config.int8_calibrator = calib
                if verbose:
                    print("INT8 calibrator enabled")
            
            if verbose:
                print(f"Precision flags set: {mode_name}")
            
            profile = builder.create_optimization_profile()
            
            if input_shape[0] == -1:
                min_batch = 1
                opt_batch = batch_size
                max_batch = batch_size
            else:
                min_batch = opt_batch = max_batch = input_shape[0]
            
            min_shape = (min_batch, input_shape[1], check_height, check_width)
            opt_shape = (opt_batch, input_shape[1], check_height, check_width)
            max_shape = (max_batch, input_shape[1], check_height, check_width)
            
            if verbose:
                print(f"Min dynamic shape: {min_shape}")
                print(f"Opt dynamic shape: {opt_shape}")
                print(f"Max dynamic shape: {max_shape}")
            
            try:
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
                if verbose:
                    print("Optimization profile set successfully")
            except Exception as e:
                print(f"Failed to set optimization profile: {e}")
                if fallback and mode_idx < len(precision_chain) - 1:
                    print(f"{mode_name} mode failed, will try next precision mode...")
                    continue
                else:
                    return None, f"Failed to set optimization profile: {e}"
            
            if verbose:
                print(f"\nStart engine building...")
            try:
                serialized_engine = builder.build_serialized_network(network, config)
                
                if serialized_engine is None:
                    print(f"{mode_name} engine build failed!")
                    if fallback and mode_idx < len(precision_chain) - 1:
                        print(f"{mode_name} mode failed, will try next precision mode...")
                        continue
                    else:
                        return None, f"{mode_name} engine build failed"
                
                engine_bytes = bytes(serialized_engine)
                
                if verbose:
                    print(f"{mode_name} engine build successful!")
                    print(f"  Serialized size: {len(engine_bytes):,} bytes ({len(engine_bytes)/(1024*1024):.2f} MB)")
                
                return engine_bytes, mode_name
                
            except Exception as e:
                last_error = str(e)
                print(f"{mode_name} engine build exception: {last_error}")
                
                if fallback and mode_idx < len(precision_chain) - 1:
                    print(f"{mode_name} mode failed, will try next precision mode...")
                    continue
                else:
                    return None, f"{mode_name} engine build failed: {last_error}"
    
    return None, f"All precision modes failed, last error: {last_error}"

def replace_suffix(file, new_suffix):
    r = file.rfind(".")
    return f"{file[:r]}{new_suffix}"

def get_precision_info(precision):
    info = {
        "int8": {
            "description": "INT8 quantization (highest performance, calibration required)",
            "suffix": "_int8.engine",
            "needs_calib": True,
            "notes": "Requires calibration data, maximum precision loss"
        },
        "fp16": {
            "description": "FP16 half precision (high performance, no calibration required)",
            "suffix": "_fp16.engine",
            "needs_calib": False,
            "notes": "No calibration needed, minimal precision loss"
        },
        "tf32": {
            "description": "TF32 mixed precision (A100/Ampere+ GPU)",
            "suffix": "_tf32.engine",
            "needs_calib": False,
            "notes": "Supported on Ampere architecture and above GPUs"
        },
        "fp32": {
            "description": "FP32 single precision (highest precision, slowest speed)",
            "suffix": "_fp32.engine",
            "needs_calib": False,
            "notes": "Full precision, no quantization loss"
        }
    }
    return info.get(precision, info["fp32"])

def main():
    args = parse_arguments()
    
    print(f"\n{'#'*80}")
    print("#" + " TensorRT Model Precision Conversion Tool ".center(78) + "#")
    print(f"{'#'*80}")
    
    onnxfile = args.onnx
    calib_data_dir = args.calib_data
    precision = args.precision.lower()
    batch_size = args.batch_size
    num_calib = args.num_calib
    
    precision_info = get_precision_info(precision)
    
    if args.output:
        engine_file = args.output
    else:
        engine_file = replace_suffix(onnxfile, precision_info["suffix"])
    
    if args.cache_file:
        calibration_cache = args.cache_file
    else:
        calibration_cache = replace_suffix(onnxfile, ".cache")
    
    print(f"\n ✅ Configuration parameters:\n")
    if precision_info["needs_calib"]:
        print(f" 🔄 Calibration data directory: {calib_data_dir}")
        print(f" 🔄 Number of calibration images: {num_calib}")
        print(f" 🔄 Calibration cache: {calibration_cache}\n")
    else:
        print(f"  Calibration data: Not required")
    print(f" ✔ ONNX model:    {onnxfile}")
    print(f" ✔ Output engine: {engine_file}")
    print(f" ✔ precision:     {precision.upper()} - {precision_info['description']}")
    print(f" ✔ Notes:         {precision_info['notes']}")
    print(f" ✔ Batch size:    {batch_size}")
    print(f" ✔ Fixed input size:         {args.check_height}x{args.check_width}")
    print(f" ✔ Workspace:                {args.workspace} MB")
    print(f" ✔ Layer precision strategy: {args.layer_precision}")
    print(f" ✔ Sparsity optimization:    {'Enabled' if args.sparsity else 'Disabled'}")
    print(f" ✔ Auto fallback:            {'Disabled' if args.no_fallback else 'Enabled'}")
    
    if not os.path.exists(onnxfile):
        print(f"\nError: ONNX file does not exist - {onnxfile}")
        return
    
    if precision == "tf32":
        try:
            cuda.init()
            device = cuda.Device(0)
            device_name = device.name().lower()
            if "a100" in device_name or "a30" in device_name or "a40" in device_name or \
               "rtx 30" in device_name or "rtx 40" in device_name or \
               "ampere" in device_name or "ada" in device_name:
                print(f"\nDetected TF32 supported GPU: {device.name()}")
            else:
                print(f"\nWarning: Current GPU ({device.name()}) may not fully support TF32")
                print("TF32 performs best on Ampere architecture and above GPUs")
        except:
            print("\nWarning: Failed to detect GPU architecture, TF32 performance may be limited")
    
    calib = None
    if precision_info["needs_calib"]:
        print(f"\n{'='*60}")
        print("Step 1: Create INT8 Calibrator")
        
        if not calib_data_dir:
            print("Error: INT8 mode requires calibration data directory (--calib_data)")
            return
        
        if os.path.exists(calib_data_dir):
            print(f"Found calibration data directory: {calib_data_dir}")
            try:
                calib = EntropyCalibrator(
                    training_data_dir=calib_data_dir,
                    cache_file=calibration_cache,
                    batch_size=batch_size,
                    num_calib_images=num_calib,
                    verbose=args.verbose
                )
                print("Calibrator created successfully")
            except Exception as e:
                print(f"Calibrator creation failed: {e}")
                if not args.no_fallback:
                    print("Will fallback to FP16 mode")
                    precision = "fp16"
                    precision_info = get_precision_info("fp16")
                    engine_file = replace_suffix(onnxfile, precision_info["suffix"])
                    calib = None
                else:
                    print("Calibrator creation failed and auto fallback disabled, exit")
                    return
        else:
            print(f"Error: Calibration data directory does not exist - {calib_data_dir}")
            if not args.no_fallback:
                print("Will fallback to FP16 mode")
                precision = "fp16"
                precision_info = get_precision_info("fp16")
                engine_file = replace_suffix(onnxfile, precision_info["suffix"])
                calib = None
            else:
                print("Calibration data directory not found and auto fallback disabled, exit")
                return
    else:
        print(f"\n{'='*60}")
        print(f"Step 1: {precision.upper()} mode, no calibrator required")
    
    print(f"\n{'='*60}")
    print(f"Step 2: Build TensorRT {precision.upper()} engine")
    
    engine_data, precision_mode = build_engine(
        onnxfile, 
        calib, 
        batch_size=batch_size,
        precision=precision,
        check_height=args.check_height,
        check_width=args.check_width,
        workspace_mb=args.workspace,
        fallback=not args.no_fallback,
        layer_precision=args.layer_precision,
        sparsity=args.sparsity,
        verbose=args.verbose
    )
    
    print(f"\n{'='*60}")
    print("Step 3: Save engine file")
    
    if engine_data:
        try:
            if precision_mode == "INT8":
                engine_file = replace_suffix(onnxfile, "_int8.engine")
            elif precision_mode == "FP16":
                engine_file = replace_suffix(onnxfile, "_fp16.engine")
            elif precision_mode == "TF32":
                engine_file = replace_suffix(onnxfile, "_tf32.engine")
            elif precision_mode == "FP32":
                engine_file = replace_suffix(onnxfile, "_fp32.engine")
            
            with open(engine_file, "wb") as f:
                f.write(engine_data)
            
            file_size = os.path.getsize(engine_file)
            print(f"Engine file saved successfully!")
            print(f"   File name: {engine_file}")
            print(f"   Actual precision: {precision_mode}")
            print(f"   File size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
            
            print(f"\nPerformance tips:")
            if precision_mode == "INT8":
                print("  • INT8 mode provides the fastest inference speed")
                print("  • Suitable for scenarios with extreme speed requirements")
            elif precision_mode == "FP16":
                print("  • FP16 mode balances speed and precision")
                print("  • Suitable for most application scenarios")
            elif precision_mode == "TF32":
                print("  • TF32 mode performs excellently on Ampere architecture GPUs")
                print("  • Provides FP32-level precision with FP16-level speed")
            elif precision_mode == "FP32":
                print("  • FP32 mode provides the highest precision")
                print("  • Suitable for scenarios with extreme precision requirements")
            
        except Exception as e:
            print(f"Failed to save engine file: {str(e)}")
    else:
        print("Engine build failed, no data to save")
        print(f"Failure reason: {precision_mode}")
    
    print(f"\n{'#'*80}")
    print("#" + " Process Completed ".center(78) + "#")
    print(f"{'#'*80}")

if __name__ == "__main__":
    try:
        cuda.init()
        device_count = cuda.Device.count()
        print(f"Detected {device_count} CUDA devices")
        for i in range(device_count):
            device = cuda.Device(i)
            print(f"  Device {i}: {device.name()}")
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        print("Please check CUDA installation and GPU driver")
        exit(1)
    
    main()
