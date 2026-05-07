# 05. Training experiments

I trained three experiments:

- baseline CNN with learning rate 0.001 and batch size 128
- tuned baseline CNN with learning rate 0.0005 and batch size 256
- enhanced VGG-style CNN with learning rate 0.001 and batch size 128

The tuning was deliberately limited. I did not run a full grid search because deeper CNN training takes time, and the main aim was to make a clear comparison between a simple baseline and a better regularised model.

All models used sparse categorical cross-entropy, Adam, early stopping, and reduce-on-plateau. I recorded training time as well as accuracy, because a better model is less convincing if the accuracy gain is very small and the training cost is much larger.
