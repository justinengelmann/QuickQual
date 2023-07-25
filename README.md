# QuickQual: Lightweight, convenient retinal image quality scoring with off-the-shelf pretrained models

## QuickQual

QuickQual is a state-of-the-art method for retinal image quality scoring, using only an off-the-shelf pretrained DL
model (DenseNet121) and a sklearn SVM as classifier. It can be used easily with very few lines of code, making it very lightweight and convenient to use.

![QuickQualOverview.png](Figures\QuickQualOverview.png)


### Results for EyeQ test set

|               |    Accuracy |          AUC |           F1 |      LogLoss |       Kappa |    QuadKappa |
|:--------------|------------:|-------------:|-------------:|-------------:|------------:|-------------:|
| MCFNet        |    0.879993 |      0.95883 |     0.860631 |     0.363168 |    0.801693 |     0.895506 |
| **QuickQual** | **0.88627** | **0.968703** | **0.867457** | **0.304923** | **0.81072** | **0.901928** |

### Code and dependencies

QuickQual only requires a pretrained DenseNet121 model and a sklearn SVM classifier. QuickQual can be used
with very few lines of code:

```python
import torch
from torchvision.transforms import functional as F
from PIL import Image
import timm
import joblib

img = Image.open('/home/justinengelmann/datastorage/kaggle/test_preprocessed/10036_left.jpeg')

model = timm.create_model('densenet121.tv_in1k', pretrained=True, num_classes=0)
model.eval().cuda()
clf = joblib.load('quickqual_dn121_512.pkl')

img = F.to_tensor(F.resize(img, 512))
img = F.normalize(img, [0.5] * 3, [0.5] * 3).cuda().unsqueeze(0)
with torch.no_grad():
    features = model(img).squeeze().cpu().reshape(1, -1)
pred = clf.predict_proba(features)  # order of class probabilities: Good, Usable, Bad
print(''.join([f'{p:.4f} ' for p in pred[0]]))  # p(Good): 0.0001 p(Usable): 0.0114 p(Bad): 0.9885 
```

Note: The pytorch image library (timm) can be replaced with torchvision, as the densenet121 model is also available
there. See the [noteboook](QuickQual_inference_example.ipynb) for more details.

## QuickQual-MEME

QuickQual MEga Minified Estimator (MEME) is an even more lightweight version of QuickQual, using the same DenseNet121
plus 10 parameters for a linear model (9 weights and one bias). QuickQual-MEME produces a one-dimensional image quality
score instead of predictions for three separate classes.

QuickQual-MEME does not even require sklearn classifier, the linear model with its 10 parameters is contained in the
code below:

```python
import torch
from torchvision.transforms import functional as F
from PIL import Image
import timm

img = Image.open('/home/justinengelmann/datastorage/kaggle/test_preprocessed/10036_left.jpeg')

model = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
model.eval().cuda()

w = torch.tensor([-1411.32, 517.09, 342.41, -707.9,
                  1442.09, -23.25, -541.64, -8.44, 5.44])
b = torch.tensor([5.18])

img = F.to_tensor(F.resize(img, 512))
img = F.normalize(img, [0.5] * 3, [0.5] * 3).cuda().unsqueeze(0)

with torch.no_grad():
    feats = model(img).squeeze().cpu().reshape(1, -1)
feats = feats[:, [71, 109, 121, 53, 55, 123, 29, 133, 84]]
pred = torch.sigmoid(feats @ w + b)
print(f'Predicted p(bad): {pred.item():.4f}')  # Predicted p(bad): 0.9386
```

![QuickQualMEME_Hist.jpeg](Figures\QuickQualMEME_Hist.jpeg)

## Acknowledgements

We thank Huazhu Fu and colleagues for releasing the EyeQ dataset and MCFNet code and results. This is an amazing
contribution to the field and without their work, QuickQual would not have been possible. Check out their
repo: https://github.com/HzFu/EyeQ