# model_prepare

### our main code
./model_prepare/scripts/clip/main_patch_vit_our.py

### reference code
./model_prepare/scripts/clip/main_patch_vit.py 

### Problem
 backdoor需要对指定特定网络层进行操作，比如 line408 if name =='head':, line571: name_list=['head',  '11','fc' ], 采用dp后调用line404, for name, module in classifer.module.named_modules(), 显示不出来module name

 比对task vectors里heads.py代码，会有build_classification_head的部分，但是pytorch_classfication没有单独抽出来这个模块，打印模型网络结构仍搞不清楚问题


