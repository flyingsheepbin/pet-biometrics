import pandas as pd


m1 = pd.read_csv('b7ns_stage2/results.csv')
m2 = pd.read_csv('swin384_stage2/results.csv')
m3 = pd.read_csv('v2_stage2/results.csv')
m4 = pd.read_csv('swin224_stage2/results.csv')
# m5 = pd.read_csv('nfnet_f3_bs64_epoch52_20_aug2_resize_multi_size_45_without_crop_without_flip_stage2_8566.csv')
val = pd.read_csv('data/test/test_data.csv')

pred1 = m1['prediction']
pred2 = m2['prediction']
pred3 = m3['prediction']
pred4 = m4['prediction']
# pred5 = m5['prediction']
val['prediction'] = (pred1 + pred2 + pred3 + pred4) / 4.0
val.to_csv("final_result.csv", index=False)
print("over!")

