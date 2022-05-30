# 第二阶段使用刚训练得推理

# b7ns
cd b7ns_stage2
python infer.py


# swin384
cd ../swin384_stage2
python infer.py

# effv2l
cd ../v2_stage2
python infer.py

# swin224
cd ../swin224_stage2
python infer.py

# 融合
cd ..
python fusion.py
