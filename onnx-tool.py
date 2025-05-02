import onnx_tool
import numpy as np

onnx_filename = '/mnt/ssd/home/dcopping/simple_bev_new/simple_bev/onnx/rgb/simple_bev_1.onnx'
# onnx_filename = '/mnt/ssd/home/dcopping/simple_bev_new/simple_bev/onnx/example/qnn_onnx_model_tst_mul.onnx'
# onnx_filename = '/mnt/ssd/home/dcopping/simple_bev_new/simple_bev/onnx/example/modified_gaussianbev_1_sim.onnx'

m = onnx_tool.Model(onnx_filename)
image_shape = (1, 6, 3, 224, 400)
m.graph.shape_infer({'data': np.zeros(image_shape)})
m.graph.profile()
experiment_path = '/mnt/ssd/home/dcopping/simple_bev_new/simple_bev/onnx/example/simple_bev_1_output.txt'
m.graph.print_node_map(experiment_path)