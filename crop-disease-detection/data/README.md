# Dataset — DiaMOS Plant Disease

## Source
**Kaggle:** https://www.kaggle.com/datasets/mohamedhanyyy/diamos-plant-disease

## Description
The DiaMOS Plant dataset contains **3,505 annotated images** of pear tree leaves and fruits
captured across a full growing season (February to July).

## Classes

| Class Label   | Image Count | Description                            |
|---------------|-------------|----------------------------------------|
| Healthy       | 700         | No disease — reference class           |
| Leaf_Spot     | 850         | Circular necrotic spots on leaves      |
| Leaf_Curl     | 780         | Upward or downward curling of edges    |
| Slug_Damage   | 620         | Irregular holes and edge damage        |
| Fruit_Disease | 555         | Disease symptoms on pear fruit         |

## Setup Instructions

1. Go to: https://www.kaggle.com/datasets/mohamedhanyyy/diamos-plant-disease
2. Click **Download** (you need a free Kaggle account)
3. Extract the ZIP file
4. Place the folders inside `data/images/` so the structure looks like:

```
data/
└── images/
    ├── Healthy/
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    ├── Leaf_Spot/
    ├── Leaf_Curl/
    ├── Slug_Damage/
    └── Fruit_Disease/
```

## Generated File
After running `src/feature_extraction.py`, a file `data/features.csv` will be created.
This CSV contains the extracted 10-dimensional feature vectors for all images and IS committed to the repo.

| Column         | Description                          |
|----------------|--------------------------------------|
| contrast       | GLCM contrast feature                |
| energy         | GLCM energy feature                  |
| homogeneity    | GLCM homogeneity feature             |
| correlation    | GLCM correlation feature             |
| red_mean       | Mean of red channel                  |
| green_mean     | Mean of green channel                |
| blue_mean      | Mean of blue channel                 |
| red_std        | Std deviation of red channel         |
| green_std      | Std deviation of green channel       |
| blue_std       | Std deviation of blue channel        |
| label          | Disease class label (string)         |
