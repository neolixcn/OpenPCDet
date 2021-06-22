import onnx
import onnxruntime
import torch
import onnx.numpy_helper

# added by huxi, load rpn config
from pcdet.pointpillar_quantize_config import load_rpn_config_json
# ========================================


config_dict = load_rpn_config_json.get_config()
onnx_model_file = config_dict["vfe_onnx_file"]

onnx_model = onnx.load(onnx_model_file)
onnx.checker.check_model(onnx_model)
#check model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



#[tensor_mat_weight] = [t for t in onnx_model.graph.initializer if t.name == "linear.weight"]
[tensor_mat_weight] = [t for t in onnx_model.graph.initializer if t.name == "14"]

[tensor_bn_gamma] = [t for t in onnx_model.graph.initializer if t.name == "norm.weight"]
[tensor_bn_beta] = [t for t in onnx_model.graph.initializer if t.name == "norm.bias"]
[tensor_bn_mean] = [t for t in onnx_model.graph.initializer if t.name == "norm.running_mean"]
[tensor_bn_var] = [t for t in onnx_model.graph.initializer if t.name == "norm.running_var"]



mat_w = onnx.numpy_helper.to_array(tensor_mat_weight)
mat_w = mat_w.transpose()
mat_w_list = list(mat_w.flatten())

bn_gamma_w = onnx.numpy_helper.to_array(tensor_bn_gamma)
bn_gamma_w_list = list(bn_gamma_w.flatten())

bn_beta_w = onnx.numpy_helper.to_array(tensor_bn_beta)
bn_beta_w_list = list(bn_beta_w.flatten())

bn_mean_w = onnx.numpy_helper.to_array(tensor_bn_mean)
bn_mean_w_list = list(bn_mean_w.flatten())

bn_var_w = onnx.numpy_helper.to_array(tensor_bn_var)
bn_var_w_list = list(bn_var_w.flatten())

result_line = ""

exported_vfe_weight_file = config_dict["vfe_exported_weight_file"]

with open(exported_vfe_weight_file, 'w') as f:
    for idx,val in enumerate(mat_w_list):
        result_line += str(val) 
        result_line += " "
    result_line += "\n"

    for idx,val in enumerate(bn_gamma_w_list):
        result_line += str(val) 
        result_line += " "
    result_line += "\n"

    for idx,val in enumerate(bn_beta_w_list):
        result_line += str(val) 
        result_line += " "
    result_line += "\n"

    for idx,val in enumerate(bn_mean_w_list):
        result_line += str(val) 
        result_line += " "
    result_line += "\n"

    for idx,val in enumerate(bn_var_w_list):
        result_line += str(val) 
        result_line += " "
        
    f.write(result_line)

    



