## Refinement Unit
This is the implementation of refinement unit presented in paper section 3.3.


<p align="center"><img src="https://s2.eksiup.com/1d48b0029160.png" width="70%" alt=""/></p>

### Quick Start
Download training and validation data from 
[GoogleDrive](https://drive.google.com/open?id=1GKaTvzTPh5xZuO3ybyK_lbERrR6C-iv3) (227 MB). 
Put it under `$(ROOT)/refiner` and unzip via `unzip data.zip`.

To validate a pretrained model, you can run:
```
python refiner/main.py --mode test --load models/h36m/refiner.pth.tar
```
For training:
```
python refiner/main.py --mode train
```