# Time Series Learning
This project is intended to implement Deep NN / RNN based solution in order to develop flexible methods that are able to adaptively fillin, backfill, and predict time-series using a large number of heterogeneous training datasets.

The perfect solution must at least exceed performance of plain vanila Random Forest Regressor which considered as scoring baseline.

## Overview
The goal of this project is to develop flexible methods that are able to adaptively fillin, backfill, and predict time-series using a large number of heterogeneous training datasets. The data is a set of thousands of aggressively obfuscated, multi-variate timeseries measurements. There are multiple output variables and multiple input variables.

For each time-series, there are parts missing. Either individual measurements, or entire sections. Each time-series has a different number of known measurements and missing measurements, the goal is to fill in the missing output variables with the best accuracy possible. How the missing input variables are treated is an open question, and is one of the key challenges to solve.

This problem, unlike many data science contest problems, is not easy to fit into the standard machine learning framework. Some reasons that this is the case:
* There are multiple time-series outputs.
* There are multiple timeseries inputs.
* The time-series are sampled irregularly, and in
different time points for each subject.
* There is a huge amount of missing data, and it
is not missing at random.
* Many of the variables are nominal/categorical,
and some of these are very high cardinality. The most important variable, subject id, is the primary example. A good solution should not ignore the subject id.

## Scoring
The score for each individual prediction p, compared against actual ground-truth value t, will be |p - t|. The score for each row, r, will then be the mean of the scores for the individual predictions on that row (possibly 1, 2, or 3 values).
Over the full n rows, your final score will be calculated as 10 * (1 - Sum(r) / n). Thus a score of 10.00 represents perfect predictions with no error at all.

## Realisation
In order to fulfill requested task was implemented two solutions based on Recurent Neural Network and Deep Learning Neural Network architectures.
It was compared performance of both against plain vanila implementation based Random Forest Regressor.

The Deep NN was found as superior to RNN for this task, but with not too big difference. But, unfortunatelly, both still lag behind Random Forest Regressor.

Scores per method:
* Deep NN: 9.861
* RNN: 9.830
* Random Forest Regressor: 9.880 (baseline)

## Best results

```
       0.026730638156987854  0.007701583490203154  0.03510046831242789
count         238897.000000         238897.000000        238897.000000
mean               0.072598              0.282652             0.195517
std                0.067135              0.157973             0.249721
min                0.000000              0.000000             0.000000
25%                0.030325              0.171160             0.001698
50%                0.053854              0.246471             0.005034
75%                0.086798              0.346608             0.400094
max                0.465094              0.891380             0.894128

60, 100, 0.05 -> 1e-8, Adagrad, RNN

epoch 59, train loss: [ 0.04069507], score: [ 9.84009604]
Validate score: 9.8366469
Test score: 9.82 (vp_30_07_19_52.csv)
________________________________________________________________________

       0.0378213827224494            0.0          0.0.1
count       238897.000000  238897.000000  238897.000000
mean             0.071949       0.279996       0.193371
std              0.067764       0.156745       0.249946
min              0.000000       0.000000       0.000000
25%              0.030075       0.169914       0.000000
50%              0.053832       0.243377       0.002575
75%              0.086068       0.345924       0.397201
max              0.503966       0.925782       0.830723


180, 100, 0.05 -> 1e-8, Adagrad, RNN

epoch 179, train loss: [ 0.04046797], score: [ 9.84038537]
Validate score: 9.8366469
Test score: 9.82 (vp_31_07_00_21.csv)

------------------------
Predictions:
            yvl1_est       yvl2_est       yvl3_est
count  238898.000000  238898.000000  238898.000000
mean        0.071048       0.278478       0.190451
std         0.068349       0.157351       0.247715
min         0.000000       0.000000       0.000000
25%         0.028766       0.166988       0.000000
50%         0.053941       0.245374       0.000350
75%         0.084177       0.340235       0.389137
max         0.501959       0.881149       0.793660

101, 100, 5e-4 -> adam, tanh, shuffle, RNN

epoch 100, train loss: [ 0.03829205], score: [ 9.88033067]
Validate score: 
Test score: 9.83 (vp_02_08_23_13.csv)

------------------------
Predictions:
            yvl1_est       yvl2_est       yvl3_est
count  238898.000000  238898.000000  238898.000000
mean        0.070911       0.278394       0.188772
std         0.066999       0.158090       0.247499
min         0.000000       0.000000       0.000000
25%         0.029590       0.166484       0.000000
50%         0.053094       0.246164       0.001593
75%         0.082912       0.340488       0.407425
max         0.507457       1.000000       0.828317

81, 100, 1e-4 -> adam 0.9/0.99, shuffle, reg1e-3, preprocessing, DeepNN[50, 20]

epoch 80, train loss: [ 0.03659411], score: [ 9.88233213]

Test score: 9.85 (vp_03_08_16_27.csv)

------------------------
Predictions:
            yvl1_est       yvl2_est       yvl3_est
count  238898.000000  238898.000000  238898.000000
mean        0.071074       0.278243       0.188506
std         0.068041       0.157917       0.247308
min         0.000000       0.000000       0.000000
25%         0.029369       0.166125       0.000000
50%         0.053656       0.246783       0.001329
75%         0.082850       0.337197       0.406893
max         0.515826       0.976951       0.892692

80, 100, 5e-5 -> adam bias 0.9/0.99, shuffle, reg1e-3, preprocessing, DeepNN[60, 30]

epoch: 79, train loss: [ 0.03626305], score: [ 9.88334681], learning rate: 5e-07

Test score: 9.85 (vp_04_08_16_40.csv)

------------------------
Predictions:
            yvl1_est       yvl2_est       yvl3_est
count  238898.000000  238898.000000  238898.000000
mean        0.070833       0.278369       0.188905
std         0.068485       0.157788       0.246909
min         0.000000       0.000000       0.000000
25%         0.028804       0.166068       0.000000
50%         0.053427       0.246008       0.003569
75%         0.083206       0.340325       0.406023
max         0.525686       0.954895       0.810934

60, 100, 5e-5, adam, reg1e-4, preprocessing DeepNN[256, 128]

epoch: 59, train loss: [ 0.03710156], score: [ 9.88219018], learning rate: 5e-07

Test score: 98.53 (vp_06_08_22_49.csv)

------------------------
Predictions:
            yvl1_est       yvl2_est       yvl3_est
count  238898.000000  238898.000000  238898.000000
mean        0.070915       0.278305       0.189096
std         0.068387       0.157991       0.247285
min         0.000000       0.000000       0.000000
25%         0.029102       0.166380       0.000000
50%         0.053139       0.245979       0.003406
75%         0.083086       0.339003       0.406160
max         0.534707       0.998148       0.841650

180, 100, 5e-5, adam, reg1e-4, preprocessing DeepNN[256, 128]

epoch: 179, train loss: [ 0.03601578], score: [ 9.88568121], learning rate: 5e-07

Test score: 98.56 (vp_07_08_21_22.csv)

------------------------
Predictions:

count        238897.000000  238897.000000  238897.000000
mean              0.071045       0.277557       0.188097
std               0.068731       0.158530       0.247959
min               0.000000       0.000000       0.000000
25%               0.029443       0.165544       0.000000
50%               0.053455       0.246995       0.000000
75%               0.082807       0.338127       0.407639
max               0.658165       0.965393       0.885541


60, 100, 5e-2, Adagrad, reg1e-4, features selected, DeepNN[128, 32]

Test score: 98.61 (vp_10_08_11_45.csv)

------------------------
Predictions:
            yvl1_est       yvl2_est       yvl3_est
count  238898.000000  238898.000000  238898.000000
mean        0.072625       0.284743       0.499098
std         0.071510       0.152186       0.184594
min         0.001185       0.014566       0.001994
25%         0.029601       0.170947       0.352773
50%         0.053907       0.249249       0.526542
75%         0.084633       0.343988       0.653956
max         0.830919       0.943561       0.883551

validation baseline - Random Forest Regressor

Test score: 98.80 (vp_tree_10_08_2016.csv)
```
## Directory structure and running
### The directories:
* 'data' directory contains training / testing data samples
* 'src' directory has source files

### The source files:
The main runners are 'src/deep_learning_runner.py' and 'src/vanila_rnn.py' for starting 'Deep NN' and 'RNN' correspondingly.
The 'src/score_validator.py' may be used to calculate score over test data saples run results.

The 'src/utils/train_validate_splitter.py' can be used in order to generate train/validate data samples for training from 'data/trainng.csv' file

## Dependencies:
* [Numpy](http://www.numpy.org)
* [Pandas](http://pandas.pydata.org)
* [scikit-learn](http://scikit-learn.org/stable/) 

## References
* [Stanford CS class CS231n](http://cs231n.github.io)
* [UFLDL Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Recurrent Neural Networks](http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/recurrentNeuralNetworks.php)
* [Generating Sequences With Recurrent Neural Networks arXiv:1308.0850](http://arxiv.org/abs/1308.0850v5)
* [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling arXiv:1412.3555](http://arxiv.org/abs/1412.3555v1)
* [Adam: A Method for Stochastic Optimization arXiv:1412.6980](http://arxiv.org/abs/1412.6980v8)
* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification arXiv:1502.01852](http://arxiv.org/abs/1502.01852v1)
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift arXiv:1502.03167](http://arxiv.org/abs/1502.03167v3)
* [RMSProp and equilibrated adaptive learning rates for non-convex optimization arXiv:1502.04390](http://arxiv.org/abs/1502.04390v1)
* [DRAW: A Recurrent Neural Network For Image Generation arXiv:1502.04623](http://arxiv.org/abs/1502.04623v2)
* [Directly Modeling Missing Data in Sequences with RNNs: Improved Classification of Clinical Time Series arXiv:1606.04130](http://arxiv.org/abs/1606.04130v1)
