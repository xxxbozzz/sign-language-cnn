# 06. Evaluation outputs

After training, the code saves a model comparison table, classification report, confusion matrix, training curves, and sample predictions.

The comparison table gives the main result. The enhanced VGG-style model reached the best test accuracy. The classification report checks whether this result is spread across letters rather than coming from only easy classes.

The confusion matrix was added because accuracy alone does not show where the model fails. In this project, the remaining errors mostly appear between visually similar signs, which is a more useful finding than simply saying the model is accurate.

The sample prediction grid was added as a final sanity check. It lets me look at actual test images and see whether the predictions make visual sense.
