# Memory-Enhanced Video Transformers: Robust Obstacle Detection for Autonomous Agricultural Rovers
![](https://github.com/TheoBiardeau/Memory-Enhanced-Video-Transformers/blob/main/VMTAD_GIF.gif)

## Data
As the dataset is private, we cannot provide it in its entirety. However, for scientific reproducibility, we are providing a sequence of 600 images representative of the dataset, please open an issue to obtain the data. The images for the qualitative analysis is avaible in the repo.

Data should be organized in the following tree structure with the exact same names:

```
└── dataset/
    │
    ├── AD/
    │   │
    │   ├── AD_1/
    │   │   ├── frame_0000.jpg
    │   │   ├── frame_0001.jpg
    │   │   └── ...
    │   │
    │   ├── AD_2/
    │   └── ...
    │
    └── AD_labels/
        │
        ├── AD_1/
        │   ├── frame_0000.jpg
        │   ├── frame_0001.jpg
        │   └── ...
        │
        ├── AD_2/
        └── ...
```

## Weights
All weights are available at the following link: https://zenodo.org/records/15295555

## Evaluation
The evaluation is performed in the notebook eval.ipynb.

## Visualization
We can perfom same visualisation via the notebook visu.ipynb
