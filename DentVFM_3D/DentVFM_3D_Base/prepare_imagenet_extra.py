from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, 
                       root="../../../Origin_Datasets/ILSVRC_2012/Data/CLS-LOC", 
                       extra="../../../Origin_Datasets/ILSVRC_2012/extra")
    dataset.dump_extra()