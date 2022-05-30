
# 第二阶段直接加载模型推理

# b7ns
cd b7ns_stage2
python infer_ckpt.py


# swin384
cd ../swin384_stage2
python infer_ckpt.py

# effv2l
cd ../v2_stage2
python infer_ckpt.py

# swin224
cd ../swin224_stage2
python infer_ckpt.py

# 融合
cd ..
python fusion.py
