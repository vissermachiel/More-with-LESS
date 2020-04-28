# Introduction {#introduction}

## Classification {#classification}

Classification is an important statistical tool that is widely used in a variety of scientific fields to determine to which of a set of categories a new observation belongs, based on example data. Examples of classification from everyday life are detection of spam in your mailbox, determining whether someone is creditworthy for a mortgage, a medical doctor assigning a diagnosis to a patient, and deciding whether [the salamander you caught in a pond in western France](https://doi.org/10.1002/ece3.2676) belongs to the species _Triturus cristatus_ or _T. marmoratus_. An algorithm or model that maps input data to a category is called a classifier. This classifier determines which of a set of variables are important to distinguish between the categories or classes. By comparing genetic information of people with a certain disease with that of people that do not have this disease it is possible to train a classifier that calculates the likelihood that a person develops this particular disease. It is therefore also possible to test which genes play an important role in the development of certain diseases, such as colon cancer. For these studies DNA microarrays are used to analyse gene expressions. Although it is difficult and costly to find and measure this for thousands of cancer patients, it is becoming easier and cheaper to measure the activity of thousands of genes or even the entire genome of one patient at once. Therefore, the number of variables (genes) in the produced datasets of these studies is multiple orders of magnitude larger than the number of samples (patients). The analysis of these so-called high-dimensional datasets brings some mathematical problems, which require adaptations to existing classification methods. This will be explained in more detail in Chapter \@ref(methods).

## Cross-validation {#cv}

Statistical classification is an example of a supervised learning task, where a classification model is learned from example data, which can then be used to make predictions for new data. The dataset that is used to train the model parameters is called the training dataset. It contains samples for which the corresponding class labels are already known. These examples are used to infer the model parameters such that the classifier assigns these samples to their correct classes. This trained model can also be used to determine the class labels for new and unseen data. In addition to the model parameters, classification methods for high-dimensional data make use of regularisation parameters (explained in Chapter \@ref(methods)). In order to select the value for this regularisation parameter that results in the best classification performance, multiple models, each with a different value for the regularisation parameter, are trained on the same training dataset. Each model is then applied on the same set of new data, called the validation dataset, which contains samples with known labels. Each model makes predictions for these validation samples and the predicted labels are compared to the true labels. The regularisation parameter that results in the model with the best predictions on the validation dataset is chosen. 

Testing the performance on the validation dataset, rather than on the training dataset, helps to avoid underfitting and overfitting. Underfitting means that the model is not sophisticated enough to capture enough patterns in the training data to make correct predictions. An underfitted model performs poorly on both the training and validation sets. Overfitting means that the model is too sophisticated in capturing patterns, such that it also captures noise from the training data. For this reason an overfitted model performs extremely well on the training dataset, but generalises very poorly to the validation dataset. A good model should avoid under- and overfitting and should perform well on both the training and the validation datasets (see Figure \@ref(fig:underfitting-overfitting)). The regularisation parameter plays an important role in this trade-off.

In order to assess how well the chosen classification model generalises to new data, it is tested on yet another unseen dataset with known labels, called the test dataset. If the samples in the test dataset are representative of the whole population, this gives a good estimate of the performance of the classifier on new data for which the labels are not yet known. When different classification methods are being compared, their performances on the test dataset can be used to decide which method performs best.

```{r underfitting-overfitting, echo=FALSE, out.width="60%", fig.cap="Model performance on the training and validation datasets in case of underfitting and overfitting. Image courtesy of @TDS:a-underfitting-overfitting."}
knitr::include_graphics("Figures/1_Introduction/Underfitting_Overfitting.png")
```

When there is enough data available the dataset can be split into three fixed parts: training, validation, and testing sets. The assignment of the data samples to each of these partitions is done randomly. The proportion of the data that should be assigned to either of these partitions depends on the total sample size of the dataset. Often 80% of the data is used for training, 10% for validation of the models, and 10% for testing of the final model. This procedure works well when the sample size of the dataset is large. Even after splitting the data into these fixed parts, there are still enough samples to properly train the classification models. However, this strategy is less appropriate for high-dimensional datasets, where the number of samples is much smaller than the number of variables. Splitting such data into fixed train, validation, and test sets reduces the amount of available data to train the model and therefor makes it harder to train these models accurately. When there are only a few samples at hand it is recommended to use cross-validation to analyse the data.

For cross-validation the dataset is randomly split into $k$ parts, called folds. One of these folds is then used as test set, while the other $(k-1)$ folds are taken together and used as training set. Once the classification model is trained on this training set, predictions are made for all samples in the test set. In the next iteration another fold is selected as test set and the model is trained on the other folds. These iterations continue until all folds have been used as test set and predictions are made for all samples. This makes it possible to assess the model performance based on all available data. Often, $k$ is chosen to be `10` and the corresponding procedure is called 10-fold cross-validation. In case there are only a few samples available in the dataset, it is common practice to set $k$ equal to $n$, the total sample size of the dataset. This means that predictions are made for each sample, based on a model that is trained on all other samples in the dataset. This is known as leave-one-out cross-validation.

```{r cv, echo=FALSE, out.width="100%", fig.cap="Schematic overview of the process of cross-validation with a nested loop for parameter tuning."}
knitr::include_graphics("Figures/1_Introduction/Cross-validation.png")
```

When analysing high-dimensional datasets or other datasets that require the optimisation of a regularisation parameter, a nested cross-validation procedure is used. This means that within each of the $k$ cross-validation iterations, the optimum value for this regularisation parameter is obtained using an inner cross-validation procedure on the $(k-1)$ training folds. For this purpose the training set is again randomly split into $m$ folds. One fold is used as validation set, while the other $(m-1)$ folds are used for model training. After the model is trained, predictions are made for the samples in the validation set. In the next iteration another fold is used as validation set and the model is trained on the other folds. This is repeated until all $m$ folds have been used for model validation and predictions are made for all samples. The average model performance is calculated across all validation folds. In order to find out which value from a pre-specified sequence of potentially appropriate regularisation values results in the best model, this inner cross-validation loop is carried out for each of these values. The regularisation value that results in the best prediction performance, averaged over all validation folds, is selected for the regularisation parameter and is used in the model training on the training set in the corresponding iteration of the outer cross-validation loop. This procedure is independently used for each of the $k$ iterations of the outer cross-validation loop. Finally, the test performance of the classification method is assessed by calculating the average performance across all $k$ test sets. A schematic overview of the process of nested cross-validation is shown in Figure \@ref(fig:cv). This procedure will be used throughout all simulation studies and experiments in this thesis (Chapters \@ref(scaling) and \@ref(simulations)).

## Performance metrics {#metrics}

There are different ways to assess the performance of a classification model. In a supervised learning setting each sample gets a predicted class label, while the actual class label is also known. Therefore, the results of a classification model can be summarised in a confusion matrix, as depicted in Figure \@ref(fig:confusion-matrix).

```{r confusion-matrix, echo=FALSE, out.width="50%", fig.cap="Confusion matrix."}
knitr::include_graphics("Figures/1_Introduction/ConfusionMatrix.png")
```

Each row of this matrix contains the predicted class labels (i.e. `Positive` and `Negative`), while the columns represent the actual class labels (also `Positive` and `Negative`). This means that there are four different scenarios possible when making class predictions for a data sample. When the class is predicted to be `Positive` and this is in accordance with the actual class, this is called a true positive (TP). However, when the class is actually `Negative`, this mistake is called a false positive (FP) or a type I error. On the other hand, when the class is predicted to be `Negative` and the actual class is also `Negative`, this is called a true negative (TN). But when it actually should have been predicted as `Positive`, this is a false negative (FN) or a type II error. The confusion matrix shows the numbers of true positives, false positives, true negatives, and false negatives made for a certain classifier on a dataset.

An intuitive way to evaluate the performance of a classification model is by looking at the proportion of correctly classified samples out of all samples. This is called the prediction accuracy. Likewise, the proportion of misclassifications is called the prediction error. Calculations of these performance metrics, based on the confusion matrix, are as follows:

$$\text{Accuracy} = \frac{{TP} + {TN}}{{TP} + {FP} + {TN} + {FN}}, \qquad \text{Error} = \frac{{FP} + {FN}}{{TP} + {FP} + {TN} + {FN}}.$$

Although the prediction accuracy and error come with easy interpretations, they have some disadvantages too. When the class distribution in the dataset is unbalanced, meaning that samples from one of the classes occur disproportionately more often than those from the other class, the accuracy and error could be misleading. Suppose we have a dataset containing 9,990 samples labelled as `Negative` and only 10 samples with label `Positive`. If a classification model predicts every sample to belong to the class with label `Negative`, then it achieves an accuracy of $0.999$ and an error of just $0.001$. This result is misleading, since the model is not able to detect any `Positive` samples. Hence, the accuracy (and error) is not a suitable metric to evaluate the performance of a classifier.

The decision on a suitable performance metric also depends on the costs that are involved in making wrong predictions. If the cost of a false positive is high, while the cost of a false negative is relatively low, then the classification model should be optimised by minimising the proportion of false positives out of all classes that are predicted to be positive (the false discovery rate). In other words, the proportion of true positives out of all classes that are predicted to be positive should be maximised. This is called the precision. An example of a scenario where precision is a good performance metric is the detection of spam in your mailbox. It is inconvenient when a spam email is not detected and gets in your inbox (false negative), but it is not a big problem. But on the other hand, when an important email is detected as spam and ends up in your spam folder (false positive), this could have bad consequences if you never check this. Other examples are pregnancy tests and deciding whether today's weather is good enough to launch a new satellite into orbit. Based on the confusion matrix, the precision is calculated as follows:

$$\text{Precision} = \frac{TP}{TP + FP}.$$

When the cost of a false negative is high, while the cost of a false positive is relatively low, then the classification model should be optimised by minimising the proportion of false negatives out of all classes that are actually positive (the false negative rate). This means that the proportion of correctly predicted classes out of all positive classes should be maximised. This is called the true positive rate (TPR), recall, sensitivity, power, or the probability of detection. An example of a scenario where recall is a good performance metric is a call centre that wants to sell a product or service. They cannot cal everyone, so they have to make a selection of potential buyers. When they decide to call someone who is not convinced to buy the product (false positive) this is not really a problem. However, when they do not call someone who would be easily convinced to buy the product, they miss out on a potential customer (false negative). Other examples are recommendation systems for webshops or the detection of a disease in a patient. Based on the confusion matrix, the recall is calculated as follows:

$$\text{Recall} = \text{TPR} = \frac{TP}{TP + FN}.$$

As can be deduced from the confusion matrix (Figure \@ref(fig:confusion-matrix)), it is not possible to maximise both precision and recall at the same time. There is always a trade-off between these two performance metrics. However, for situations when both precision and recall are important, there is another performance metric available, called the $F_1$ score. This is the harmonic mean between precision and recall and is calculated as follows:

$$F_1 \text{ score} = \left( \frac{\text{precision}^{-1} + \text{recall}^{-1}}{2} \right)^{-1} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.$$

Another performance metric is the false positive rate (FPR), also called the fall-out. This is the proportion of actual negatives that are wrongly predicted to be positive:

$$\text{Fall-out} = \text{FPR} = \frac{FP}{FP + TN}.$$
    
The true positive rate (TPR, recall) and the false positive rate (FPR, fall-out) can be combined into a single metric for the prediction performance of a classification model. In order to do this, the threshold for deciding whether a sample is classified as `Positive` or `Negative` is increased in small steps (e.g. `0.00`, `0.01`, `0.02`, $\ldots$, `1.00`). For each of these threshold values a confusion matrix can be produced, from which the TPR and FPR can be calculated. These values are then plotted in a single graph, with FPR on the horizontal $x$-axis and TPR on the vertical $y$-axis. The resulting probability curve of all combinations of FPR and TPR for all threshold values is called the Receiver Operating Characteristic (ROC) curve (see Figure \@ref(fig:auroc)). When the true positive rate is equal to the false positive rate, the ROC curve is a diagonal line. This means that the proportion of correctly classified `Positive` samples is the same as the proportion of incorrectly classified samples that are `Negative`. Based on the ROC curve it is possible to choose the optimal threshold value for separating the classes. This choice depends on the amount of false positives you are willing to accept. In order to compare the performance of different classification methods, their corresponding ROC curves can be compared by assessing the area under the curve, called the AUC value. The AUC can take values between $0.5$ and $1$, where a value of $0.5$ indicates that the classifier performs not better than random guessing, while a value of $1$ means that the classifier can separate the classes perfectly. The classification method that produces the highest AUC value is considered to perform best. In this thesis the AUC value will be used as performance metric to compare different classification methods.

```{r auroc, echo=FALSE, out.width="40%", fig.cap="Schematic example of an ROC curve (green line) following all combinations of FPR and TPR for all threshold values. The area under the ROC curve (grey area) is called the AUC value. The diagonal line shows the ROC curve of random guessing. Image courtesy of @TDS:b-auroc"}
knitr::include_graphics("Figures/1_Introduction/AUROC.png")
```

## Outline of this thesis {#outline}

The focus of this thesis lies on two-class linear classification problems for high-dimensional data, where the number of variables is orders of magnitude larger than the number of samples (e.g. 10 to 100 samples with measurements on 1,000 to 10,000 variables). A good classification method for such data should result in a good prediction performance, while using as few variables as possible. The classification method proposed by @VeenmanTax:LESS, called LESS (Lowest Error in a Sparse Subspace), fulfils both of these criteria and supposedly outperforms comparable methods [@VeenmanTax:LESS; @VeenmanBolck:LESSmulti]. In this thesis we take a closer look at this classification method and compare its performance to three other popular methods for the classification of high-dimensional data: linear regression with $\ell_1$ regularisation, logistic regression with $\ell_1$ regularisation, and support vector machine with $\ell_1$ regularisation. Chapter \@ref(methods) of this thesis will explain the details of these four methods.

The success of the LESS classifier could be ascribed to the estimated model of the class distributions, which scales the data using distances to the known class means of each variable. In Chapter \@ref(scaling) we take a closer look at this variable scaling method. Two variations of this scaling method are described, which in addition to the class means also make use of the variance of the classes. These three LESS scaling methods are combined with all four classification methods. The performance of these classifiers will be compared to the performance of the same classification methods (except for LESS) in combination with regular standardisation and when there is no data scaling at all. So in total there are 18 different classifiers considered in this study.

In Chapter \@ref(simulations) a number of simulation studies are carried out in a variety of scenarios to provide more insight into the differences in performance between the four considered classification methods in combination with the five scaling methods. The first simulation study demonstrates what happens when the number of non-meaningful variables with Gaussian noise increases. The second simulation study shows what happens to the performance when we change the variance of the noise variables. In the third simulation study the ability to select relevant variables in the model is tested, while rotating the datapoints of the first two dimensions counterclockwise around the origin. This increases the correlation between these variables. Simulation 4 tests how well the classification methods perform when the sample size of the training dataset becomes very small. The final simulation study tests the robustness of the classifiers to non-Gaussian noise. The same procedure of the first simulation is followed, but instead of the normal distribution we make use of the Cauchy distribution to generate noise variables. This distribution contains more mass in the tails than a Gaussian distribution, such that there is more overlap between the datapoints of the two classes. For all simulation studies the classification performances of the different classifiers are being compared by looking at the model dimensionality and the AUC value on the test dataset.

Next to the simulation studies, we also compare the performance of the proposed classifiers on eight real-world high-dimensional datasets from a variety of research fields in Chapter \@ref(experiments). In addition to the analyses of these datasets, we run an experiment to test what happens with the classification performance when the sample sizes of the datasets are reduced, following the same procedure as for simulation 4.

Chapter \@ref(results) contains the results and discussion of the simulation studies and analyses on the real-world datasets. The performance of the four different classification methods, as well as the performance of the five different variable scaling methods are being compared with one another. In addition, we aim to provide a theoretical explanation of the differences between the classification methods, as well as between the different types of scaling. With these results we can give recommendations for selecting the best classification method in combination with the optimal variable scaling method for a number of different scenarios. The conclusions are summarised in Chapter \@ref(conclusion). 

All plots resulting from the simulation studies and experiments on real datasets are put together in an interactive dashboard, which can be downloaded from GitHub: <https://github.com/vissermachiel/More-with-LESS>. The dashboard with interactive plots makes it possible to (de)select each classifier in these figures in order to better compare a subset of certain classifiers of interest. The R code of the LESS classifier and the functions for the discussed variable scaling methods can also be found in this online repository. After this master thesis is finished, we aim to combine all code into an R package.