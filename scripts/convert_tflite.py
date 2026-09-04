#!/usr/bin/env python3
"""
Convert PyTorch model to TensorFlow Lite for mobile deployment.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class MobileNetV1(nn.Module):
    """MobileNet V1 architecture."""
    def __init__(self, width_mult=1.0, num_classes=1000):
        super().__init__()
        input_channels = 32

        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(),
        )

        self.blocks_config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.blocks = nn.ModuleList()
        self.last_channels = input_channels

        for t, c, n, s in self.blocks_config:
            output_channels = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(self._make_block(input_channels, output_channels, stride))
                input_channels = output_channels
                self.last_channels = output_channels

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_channels, num_classes)

    def _make_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def convert_pytorch_to_tflite(model, sample_input, quantize=False, float16=False):
    """Convert a PyTorch model to TFLite format."""
    # Trace the model
    traced = torch.jit.trace(model, sample_input)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model_string(
        # Use from_py_func approach instead
    )

    # Actually, use the concrete function approach
    import tensorflow as tf

    @tf.function
    def model_fn(x):
        return model(x)

    concrete_fn = model_fn.get_concrete_function(
        tf.keras.layers.Input(sample_input.shape[1:], batch_size=1)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_fn],
        input_shapes={0: sample_input.shape}
    )

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if float16:
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    return tflite_model


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model (.pth)")
    parser.add_argument("--output", type=str, default="models/model.tflite", help="Output TFLite path")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic range quantization")
    parser.add_argument("--float16", action="store_true", help="Use float16 precision")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")

    args = parser.parse_args()

    # Load PyTorch model
    print(f"Loading model from {args.model}...")
    model = MobileNetV1(width_mult=1.0)
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PyTorch model parameters: {total_params / 1e6:.2f}M")

    # Create sample input
    sample_input = torch.randn(1, 3, args.input_size, args.input_size)

    # Convert to TFLite
    print("Converting to TensorFlow Lite...")
    try:
        tflite_model = convert_pytorch_to_tflite(
            model, sample_input, quantize=args.quantize, float16=args.float16
        )
    except Exception as e:
        print(f"Conversion error: {e}")
        print("Falling back to basic conversion...")
        # Fallback approach
        tflite_model = None

    # Save TFLite model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tflite_model is not None:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        tflite_size = len(tflite_model) / 1024
        print(f"\nTFLite model saved to {output_path}")
        print(f"TFLite model size: {tflite_size:.1f} KB")

        # Verify with tflite-runtime
        from tflite_runtime.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(output_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")

        # Test inference
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_data = np.array([np.random.randn(*input_details[0]['shape']).astype(np.float32)])
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data - zero_point) * scale
        else:
            input_data = np.array([np.random.randn(*input_details[0]['shape']).astype(input_dtype)])

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        print(f"Test inference output shape: {output_data.shape}")
        print(f"Test inference output: {output_data}")
    else:
        print("TFLite conversion failed - model not saved")


if __name__ == "__main__":
    main()