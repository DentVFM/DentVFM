from monai.data import Dataset

class ST_Lung(Dataset):
    def __init__(self, data, transform):
        super().__init__(data=data, transform=None)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])
    
class ST_Lung_eval(Dataset):
    pass