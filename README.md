# 3D_Siamese Network
paper:Improving Text Semantic Similarity Modeling through a 3D Siamese Network
# Environment
+ python3.10
+ tensorflow2.10
+ cuda11.6
+ transformers4.17
+ RTX3090

python train.py

# Train with your data
Process your own dataset into a csv file with three fields for header sentence1, sentence2, label, as shown below:

![image](https://github.com/843ECAI2023/3D_SN/assets/137853293/8788f0e7-e035-42c4-ad10-bf60e3a1a5e2)

Split the data into train.csv, dev.csv, test.csv and put them in . /inputs/dataset folder.


