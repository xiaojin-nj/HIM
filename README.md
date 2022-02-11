# HIM
Hybrid Interest Modeling withLimited Behavior

# Public data
We will expose the data set at the appropriate time

# Prepare
## Requirements
* python 2.7
* Tensorflow >= 1.0.4

## Data preprocess
The dataset is Amozon music or electronics data, download the data and it is split into training set(70%), validatation set (10%), and testing set (20%)
```
sh data.sh
```

# Model
The ./model/hhin_model.py is the model file for HHIN. Train or test model for HHIN.
```
cd ./tda_core
python tda_runner.py running_mode=train
python tda_runner.py running_mode=test
```

# Citation
If you find this code useful in your research, please cite:

```

```

# Acknowledgements
