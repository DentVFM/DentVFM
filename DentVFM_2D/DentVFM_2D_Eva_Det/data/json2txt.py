import json
from pathlib import Path


prefix = Path('Osteolysis_Carotis')
root = prefix / 'annotations'

for ann_file in root.glob('*.json'):
    with open(ann_file, 'rb') as f:
        ann = json.load(f)

    for subject, ann in ann.items():
        ann_b = ann['bounding_boxes']
        for cls in ann_b:
            if cls != 'carotis_plaques':
                continue
            
            ann_c = ann_b[cls]
            for key in ann_c:
                ann_k = ann_c[key]['99']
                for bbox in ann_k:
                    bbox = map(str, bbox['box'])
                    with open(root / f'{subject}.txt', 'a') as f:
                        f.write(' '.join([cls, *bbox]) + '\n')
