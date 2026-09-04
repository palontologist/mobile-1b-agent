#!/usr/bin/env python3
"""
LiteRT Inference App for 1B Model on Mobile

Run TFLite inference for the converted model using tflite-runtime.
"""
import argparse
import numpy as np
import sys
from pathlib import Path
from tflite_runtime.interpreter import Interpreter


def run_inference(tflite_path, input_data, use_float16=False):
    """Run inference with a TFLite model using tflite-runtime."""
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model: {tflite_path}")
    print(f"Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
    
    # Set input
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float16 and use_float16:
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float16))
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="LiteRT Inference for 1B Model")
    parser.add_argument("--model", type=str, required=True, help="Path to TFLite model")
    parser.add_argument("--input", type=str, required=True, help="Path to input numpy file or random")
    parser.add_argument("--float16", action="store_true", help="Use float16 precision")
    
    args = parser.parse_args()
    
    tflite_path = args.model
    
    # Load input data
    if args.input == "random":
        # Create random input matching expected shape
        interpreter_test = Interpreter(model_path=tflite_path)
        interpreter_test.allocate_tensors()
        input_details = interpreter_test.get_input_details()
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        input_data = np.random.randn(*input_shape).astype(input_dtype)
        print(f"Generated random input with shape {input_shape}")
    elif args.input.endswith('.npy'):
        input_data = np.load(args.input)
    else:
        print("Error: Input must be a .npy file or 'random'")
        sys.exit(1)
    
    # Ensure input has batch dimension
    if len(input_data.shape) == 3:
        input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    print(f"\nRunning inference with float16={args.float16}...")
    output = run_inference(tflite_path, input_data, use_float16=args.float16)
    
    print(f"\nInference complete!")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Print model size
    import os
    model_size = os.path.getsize(tflite_path)
    print(f"\nModel file size: {model_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()