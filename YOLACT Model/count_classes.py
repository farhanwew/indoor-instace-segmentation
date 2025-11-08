class_names = ('backpack', 'bed', 'blanket', 'book', 'bottle',
         'bowl', 'box', 'cabinet-merged', 'cardboard', 'ceiling-merged',
         'cell phone', 'chair', 'clock', 'couch', 'counter', 'cup', 'curtain',
         'dining table', 'door-stuff', 'fire-extinguisher', 'floor-other-merged',
         'floor-wood', 'flower', 'food-other-merged', 'fork', 'hair drier',
         'handbag', 'keyboard', 'knife', 'laptop', 'light', 'microwave',
         'mirror-stuff', 'mouse', 'oven', 'pan', 'paper', 'paper-merged',
         'person', 'pillow', 'plate', 'plug', 'potted plant', 'refrigerator',
         'remote', 'rug-merged', 'scissor', 'scissors', 'shelf', 'sink', 'spoon',
         'st-', 'stairs', 'stove-merged', 'table-merged', 'teddy bear', 'toaster',
         'toilet', 'toothbrush', 'towel', 'toy', 'trash-bin', 'tree-merged', 'tv',
         'vase', 'wall', 'wall-brick', 'wall-other-merged', 'wall-stone', 'wall-tile',
         'wall-wood', 'wallet', 'window-blind', 'window-other')

print(f"Number of classes: {len(class_names)}")
print(f"Number of classes + background: {len(class_names) + 1}")

for i, cls in enumerate(class_names, 1):
    print(f"{i:2d}: {cls}")