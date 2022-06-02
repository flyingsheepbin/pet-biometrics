# step 1:训练生成伪标签模型
cd pseudo_produce
python train.py
python pseudo.py

# step 2:训练一阶段四个模型

# model1: b7ns
cd ../b7ns_stage1
python train.py

# model2: swin384
cd ../swin384_stage1
python train.py 

# model3: effv2-large
cd ../v2_stage1
python train.py

# model4: swin224
cd ../swin224_stage1
python train.py


# step3:训练二阶段的四个模型

# model1: b7ns
cd ../b7ns_stage2
python train.py 

# model2: swin384
cd ../swin384_stage2
python train.py 

# model3: effv2-large
cd ../v2_stage2
python train.py

# model4: swin224
cd ../swin224_stage2
python train.py  
