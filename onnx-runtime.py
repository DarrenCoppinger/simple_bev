import os
import torch
import onnx
import onnxruntime as rt
import numpy as np

# Ensure the file exists
onnx_file = '/mnt/ssd/home/dcopping/simple_bev_new/simple_bev/onnx/rgb/simple_bev_1.onnx'
if not os.path.exists(onnx_file):
    raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

# Session Options
sess_options = rt.SessionOptions()
sess_options.enable_profiling = True

# Load and Check the ONNX Model
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)

# Dummy data for testing
input_shape = (1, 6, 3, 224, 400)
# device = 'cpu'
device='cuda:0'
dummy_rgb_camXs = torch.randn(*input_shape, device=device).cpu().numpy()
dummy_pix_T_cams = torch.randn(input_shape[0], input_shape[1], 4, 4, device=device).cpu().numpy()
dummy_cam0_T_camXs = torch.randn(input_shape[0], input_shape[1], 4, 4, device=device).cpu().numpy()
test_data = [(dummy_rgb_camXs, dummy_pix_T_cams, dummy_cam0_T_camXs, 0)]
classes = ["class1", "class2"]

# Run Inference
rgb_camXs, pix_T_cams, cam0_T_camXs, y = test_data[0]

input_feed = {'rgb_camXs': rgb_camXs,
              'pix_T_cams': pix_T_cams,
              'cam0_T_camXs': cam0_T_camXs}
# ort_sess = rt.InferenceSession(onnx_file, sess_options=sess_options)#
ort_sess = rt.InferenceSession(onnx_file, sess_options=sess_options, providers=['CUDAExecutionProvider'])

outputs = ort_sess.run(None, input_feed)

# Get the profiling output
profile_file = ort_sess.end_profiling()
print(f"Profiling data saved to: {profile_file}")

