# model_prepare

### our main code
./model_prepare/scripts/clip/main_patch_vit_our.py

### reference code
./model_prepare/scripts/clip/main_patch_vit.py 

### Problem
分别尝试了对
（1）head(zeroshot_weights)维度为（10,512）

（2）post_layernorm(768），存在维度匹配问题)，无法求得对应tar index，line442报错

（3）encoder.layers[-1].self_attn.out_proj.weight，维度为768,768,跑通结果
mnist上， ACC:97.5 ASR：98


但是evaluate4backdoor加载模型超参时报错

  File "/data/home/yangjinluan/conda/concrete/lib/python3.11/site-packages/torch/nn/modules/module.py", line 2152, in
load_state_dict
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for HFCLIPClassifier:



