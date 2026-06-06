torch.aoを使用し, 推論での量子化及び量子化を意識した学習を行う
https://pytorch.org/blog/quantization-aware-training/
https://github.com/pytorch/ao/



以下のチュートリアルを参照した
https://docs.pytorch.org/tutorials/unstable/gpu_quantization_torchao_tutorial.html

bf16でやる：torch.bfloat16でmodel = model.to(torch.bfloat16), image=image.to(torch.bfloat16)


Next, let's apply quantization. Quantization for GPUs comes in three main forms in torchao which is just native pytorch+python code. This includes:
int8 dynamic quantization
int8 weight-only quantization
int4 weight-only quantization

それぞれtorchaoの以下のAPIで実行できる
Int8DynamicActivationInt8WeightConfig(), Int8WeightOnlyConfig() or Int4WeightOnlyConfig()