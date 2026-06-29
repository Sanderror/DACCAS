IN THIS DIRECTORY YOU SHOULD PLACE THE DATASETS IF YOU WANT TO PERFORM RETRAINING OF THE CLASSIFIER MODEL BECAUSE YOU ADDED A NEW SOLVER. 
  
THE DATASETS CAN CAN BE OBTAINED FROM https://drive.google.com/drive/folders/1YzZnclxBOYKRuB7Gi-sbk9AqFyzhY9dT?usp=drive_link

MAKE SURE THE FOLLOWING DATA SET-UP IS ACHIEVED:

```
data/
├── text/    # Find in folder text
│   ├── Gregwar/
│   ├── Mobicms/
│   ├── King/
│   └── Wang/    # Find at location: https://drive.google.com/drive/folders/1tgefqBsNUESpgERgSP21Dn1fadrUrGUf
│       ├── 360/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── 360_gray/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── alipay/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── apple/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── baidu/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── baidu_blue/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── baidu_red/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── jd/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── jd_grey/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── jd_white/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── ms/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── qqmail/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── sina/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── weibo/
│       │   ...    # The train val test split is repeated for all these folders by default
│       ├── wiki/
│       │   ...    # The train val test split is repeated for all these folders by default
│       └── synthetic/
│           ...    # The train val test split is repeated for all these folders by default
├── moving_window/    # Find in folder moving window
│   ├── addyrus_images/
│   └── darkmarketreloaded_images/
├── image_rotation/    # Find in folder image rotation / default
│   ├── default/
│   │   ├── Pitch/
│   │   └── Dread/
│   └── special/    # Find in folder image rotation / special
├── open_circle/    # Find in folder open circle
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/   # This is not used for classification, but standard included in the export
│   └── dataset.yaml   # This is not used for classification, but standard included in the export
└── classifier/    # Find in folder classifier
    ├── odd_one_out_v1/
    ├── scrambled_image_v1/
    ├── scrambled_image_v2/
    ├── security_question_v1/
    ├── security_question_v2/
    ├── math/
    ├── minority_group/
    └── recaptchav2/  # These are obtained from https://www.kaggle.com/datasets/mikhailma/test-dataset
        ├── Bicycle/
        ├── Bridge/
        ...    # Multiple other categories are exported by default
```
