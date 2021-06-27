# SSAKT

Sequential Self-Attentive model for Knowledge Tracing.
Implemented in PyTorch 1.5.0.

## Original Datasets

[ASSISTment2009](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010)
[ASSISTment2017](https://sites.google.com/view/assistmentsdatamining/)
[EdNet](https://github.com/riiid/ednet)
[JunyiAcademy](https://www.kaggle.com/junyiacademy/learning-activity-public-dataset-by-junyi-academy)

## Train with question information

```bash
python train.py --pmodel --dataset=assist2009
```

## train without question information

```bash
python train.py --dataset=assist2009
```
