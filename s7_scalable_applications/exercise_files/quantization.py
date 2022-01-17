import torch.nn as nn
import torch.quantization
import time
'''
tensor = torch.rand(10)*100

tensor_quant_s1 = torch.quantize_per_tensor(tensor,0.1,10,torch.quint8)
tensor_quant_s10 = torch.quantize_per_tensor(tensor,0.1,10,torch.quint8)

tensor_dequant = torch.dequantize(tensor_quant_s1)
'''

import torch

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

# create a model instance
model_fp32 = M()
# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

# run the model
input_fp32 = torch.randn(1000, 4, 4, 4)
start = time.time()
res = model_int8(input_fp32)
end = time.time()
print(end-start)

start = time.time()
res_fp32 = model_fp32(input_fp32)
end = time.time()
print(end-start)

print('Stop')


