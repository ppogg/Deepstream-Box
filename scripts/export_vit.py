import os
import re
import torch
import torch.nn as nn
import onnx
import onnxsim

dynamic = False

ARCH_CONFIGS = {
    'tiny':  {'embed_dim': 96,  'depths': [2, 2, 6, 2],   'num_heads': [3, 6, 12, 24], 'def_win': 7},
    'small': {'embed_dim': 96,  'depths': [2, 2, 18, 2],  'num_heads': [3, 6, 12, 24], 'def_win': 7},
    'base':  {'embed_dim': 128, 'depths': [2, 2, 18, 2],  'num_heads': [4, 8, 16, 32], 'def_win': 12},
    'large': {'embed_dim': 192, 'depths': [2, 2, 18, 2],  'num_heads': [6, 12, 24, 48], 'def_win': 12},
}

class SwinWithPost(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.swin = swin_model
    
    def forward(self, x):
        logits = self.swin(x)
        probs = torch.softmax(logits, dim=1)
        max_probs, max_ids = torch.max(probs, dim=1)
        return torch.stack([max_ids.float(), max_probs], dim=1)

def parse_filename(filename):
    name = filename.lower()
    arch = None
    for key in ARCH_CONFIGS:
        if f"swin_{key}" in name:
            arch = key
            break
    if not arch:
        raise ValueError(f"无法识别模型架构: {filename}")
    
    win_search = re.search(r'window(\d+)', name)
    window_size = int(win_search.group(1)) if win_search else ARCH_CONFIGS[arch]['def_win']
    
    img_search = re.search(r'(\d{3})(?=\.pth|_22k|_1k|to1k)', name)
    img_size = int(img_search.group(1)) if img_search else (384 if arch in ['base', 'large'] else 224)

    num_classes = 21841 if '22k' in name else 1000
    
    print(f"配置: Arch={arch}, Win={window_size}, Img={img_size}, Classes={num_classes}")
    return arch, window_size, img_size, num_classes

def export_swin(weights_path, onnx_path=None):
    filename = os.path.basename(weights_path)
    print(f"处理: {filename}")
    
    arch_key, window_size, img_size, num_classes = parse_filename(filename)
    arch_params = ARCH_CONFIGS[arch_key]
    
    from models.swin_transformer import SwinTransformer as create_model
    model = create_model(
        img_size=img_size,
        patch_size=4,
        window_size=window_size, 
        num_classes=num_classes,
        embed_dim=arch_params['embed_dim'],
        depths=arch_params['depths'],
        num_heads=arch_params['num_heads'],
    )
    
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    model = SwinWithPost(model)
    
    if onnx_path is None:
        onnx_path = filename.replace(".pth", "_post.onnx")
    
    dummy_input = torch.randn(1, 3, img_size, img_size)
    print(f"导出: {onnx_path}")
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=17,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic else None
    )
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    model_sim, check = onnxsim.simplify(onnx_model)
    onnx.save(model_sim, onnx_path)
    
    print("完成\n")

if __name__ == "__main__":
    export_swin("weights/swin_tiny_patch4_window7_224.pth")