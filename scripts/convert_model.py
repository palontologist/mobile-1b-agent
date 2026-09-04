import torch
import torch.nn as nn
import tensorflow as tf
from pathlib import Path
import numpy as np

class Block(nn.Module):
    """Depthwise separable convolution block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_dw = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.conv_pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        return x

class MobileNetV1(nn.Module):
    """MobileNet V1 architecture aiming for ~1B parameters with width multiplier."""
    def __init__(self, width_mult=1.0, num_classes=1000):
        super().__init__()
        input_channels = 32
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(),
        )
        
        # Blocks configuration: [out_channels, repeat, stride]
        self.blocks_config = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        self.blocks = nn.ModuleList()
        self.out_channels = []
        
        for t, c, n, s in self.blocks_config:
            output_channels = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(Block(input_channels, output_channels, stride))
                input_channels = output_channels
                self.out_channels.append(output_channels)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def convert_to_tflite(model_path, tflite_path, quantize=True):
    """Convert PyTorch model to TensorFlow Lite."""
    # Load model
    model = MobileNetV1(width_mult=1.0)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Convert using TensorFlow Lite TFLite converter
    # First export to TorchScript
    scripted_model = torch.jit.trace(model, sample_input)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_py torch.jit traced(scripted_model)
    
    if quantize:
        # Dynamic range quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_path}")
    print(f"TFLite model size: {len(tflite_model) / 1024:.1f} KB")
    
    # Verify
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test inference
    input_data = np.array([np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
    print(f"Test inference output: {output_data}")
    
    return model, interpreter

if __name__ == "__main__":
    # Create and save a small model first
    print("Creating MobileNetV1 model...")
    model = MobileNetV1(width_mult=1.0)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # Save PyTorch model
    torch.save(model.state_dict(), 'models/pytorch_model.pth')
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    convert_to_tflite('models/pytorch_model.pth', 'models/model.tflite', quantize=True)