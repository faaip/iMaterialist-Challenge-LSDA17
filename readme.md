# iMaterialist Challenge (Furniture)
For Large-Scale data analysis '18 at Copenhagen University.

### Contents:
* scripts for downloading, uncompressing and also removing all image files except a few (use at own risk!!)
* train.py - calculates bottleneck features and model weights for top layer.
* predict.py - iterates through test.json, predicts for all files and random guesses for missing files.

### Tensorboard:
Training of top models can be visualized using Tensorboard. Run like this:
```
tensorboard --logdir=iMaterialist-Challenge-LSDA17-/Graph
```

### TODO:
- [x] Implement Tensorboard.
- [x] Plot of loss-error and accuracy.
- [ ] Save model.
- [ ] Implement early stopping.
- [ ] Generalize script.
- [ ] Possibility to save and load entire model.
- [ ] CLI flags for dir und so.
- [ ] Compare several pretrained models and choose the best-performing.

