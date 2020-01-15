rm(list = ls())

# Load packages
library(foreach)
library(parallel)
library(iterators)
library(doParallel)
library(glmnet)
library(lpSolve)
library(pracma)
library(ggplot2)
library(pROC)
library(R.matlab)
library(LiblineaR)
library(telegram)
library(latex2exp)
library(scales)
source("Scripts/lesstwoc_mv.R")
source("Scripts/Telegram.R")

# Load data
mydata <- readMat("Data/glioma.mat")
mydata <- cbind(as.data.frame(mydata$a$data), y = factor(mydata$a$nlab))
dim(mydata) # 50, 4435
table(mydata[, ncol(mydata)]) # 14, 7, 14, 15

# Response variable contains 4 categories >>> 1+3 vs 2+4
mydata[, ncol(mydata)] <- ifelse(mydata[, ncol(mydata)] == 1 | mydata[, ncol(mydata)] == 3, 0, 1)
table(mydata[, ncol(mydata)]) # 28, 22
mydata[, ncol(mydata)] <- as.factor(mydata[, ncol(mydata)])
lab1 <- levels(mydata[, ncol(mydata)])[1] # "0"
lab2 <- levels(mydata[, ncol(mydata)])[2] # "1"
row.names(mydata) <- 1:nrow(mydata)

ncores <- 50   # number of parallel cpu cores
K <- 100       # number of repetitions to average over
nfolds <- 10   # number of folds for results
nfolds.cv <- 4 # number of folds for cross-validation for hyper parameter
nsamples <- 10 # number of different sample sizes to test

# n1 <- n2 <- seq(3, 19, by = 2)
N <- round(pracma::logseq(8, floor(0.9 * (nrow(mydata) - 1)), n = nsamples)) 
n1 <- round(as.numeric(table(mydata[, ncol(mydata)])[1]) / nrow(mydata) * N)
n2 <- round(as.numeric(table(mydata[, ncol(mydata)])[2]) / nrow(mydata) * N)

# Create list for results [nsamples, nfolds, K]
sim.list <- list(C.cv.less.none = matrix(NA, nfolds, nsamples),
                 C.cv.less.std = matrix(NA, nfolds, nsamples),
                 C.cv.less.less = matrix(NA, nfolds, nsamples),
                 C.cv.less.lessstd = matrix(NA, nfolds, nsamples),
                 C.cv.less.lessstd2 = matrix(NA, nfolds, nsamples),
                 
                 C.cv.svml1.none = matrix(NA, nfolds, nsamples),
                 C.cv.svml1.std = matrix(NA, nfolds, nsamples),
                 C.cv.svml1.less = matrix(NA, nfolds, nsamples),
                 C.cv.svml1.lessstd = matrix(NA, nfolds, nsamples),
                 C.cv.svml1.lessstd2 = matrix(NA, nfolds, nsamples),
                 
                 numbetas.less.none = matrix(NA, nfolds, nsamples),
                 numbetas.less.std = matrix(NA, nfolds, nsamples),
                 numbetas.less.less = matrix(NA, nfolds, nsamples),
                 numbetas.less.lessstd = matrix(NA, nfolds, nsamples),
                 numbetas.less.lessstd2 = matrix(NA, nfolds, nsamples),
                 numbetas.svml1.none = matrix(NA, nfolds, nsamples),
                 numbetas.svml1.std = matrix(NA, nfolds, nsamples),
                 numbetas.svml1.less = matrix(NA, nfolds, nsamples),
                 numbetas.svml1.lessstd = matrix(NA, nfolds, nsamples),
                 numbetas.svml1.lessstd2 = matrix(NA, nfolds, nsamples),
                 numbetas.lrl1.none = matrix(NA, nfolds, nsamples),
                 numbetas.lrl1.std = matrix(NA, nfolds, nsamples),
                 numbetas.lrl1.less = matrix(NA, nfolds, nsamples),
                 numbetas.lrl1.lessstd = matrix(NA, nfolds, nsamples),
                 numbetas.lrl1.lessstd2 = matrix(NA, nfolds, nsamples),
                 numbetas.lasso.none = matrix(NA, nfolds, nsamples),
                 numbetas.lasso.std = matrix(NA, nfolds, nsamples),
                 numbetas.lasso.less = matrix(NA, nfolds, nsamples),
                 numbetas.lasso.lessstd = matrix(NA, nfolds, nsamples),
                 numbetas.lasso.lessstd2 = matrix(NA, nfolds, nsamples),
                 
                 preds.less.none = matrix(NA, nrow(mydata), nsamples),
                 preds.less.std = matrix(NA, nrow(mydata), nsamples),
                 preds.less.less = matrix(NA, nrow(mydata), nsamples),
                 preds.less.lessstd = matrix(NA, nrow(mydata), nsamples),
                 preds.less.lessstd2 = matrix(NA, nrow(mydata), nsamples),
                 preds.svml1.none = matrix(NA, nrow(mydata), nsamples),
                 preds.svml1.std = matrix(NA, nrow(mydata), nsamples),
                 preds.svml1.less = matrix(NA, nrow(mydata), nsamples),
                 preds.svml1.lessstd = matrix(NA, nrow(mydata), nsamples),
                 preds.svml1.lessstd2 = matrix(NA, nrow(mydata), nsamples),
                 preds.lrl1.none = matrix(NA, nrow(mydata), nsamples),
                 preds.lrl1.std = matrix(NA, nrow(mydata), nsamples),
                 preds.lrl1.less = matrix(NA, nrow(mydata), nsamples),
                 preds.lrl1.lessstd = matrix(NA, nrow(mydata), nsamples),
                 preds.lrl1.lessstd2 = matrix(NA, nrow(mydata), nsamples),
                 preds.lasso.none = matrix(NA, nrow(mydata), nsamples),
                 preds.lasso.std = matrix(NA, nrow(mydata), nsamples),
                 preds.lasso.less = matrix(NA, nrow(mydata), nsamples),
                 preds.lasso.lessstd = matrix(NA, nrow(mydata), nsamples),
                 preds.lasso.lessstd2 = matrix(NA, nrow(mydata), nsamples),
                 
                 preds.labs.less.none = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.less.std = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.less.less = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.less.lessstd = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.less.lessstd2 = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.svml1.none = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.svml1.std = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.svml1.less = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.svml1.lessstd = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.svml1.lessstd2 = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lrl1.none = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lrl1.std = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                  V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lrl1.less = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lrl1.lessstd = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                      V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lrl1.lessstd2 = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lasso.none = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lasso.std = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                   V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lasso.less = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                    V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lasso.lessstd = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                       V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 preds.labs.lasso.lessstd2 = data.frame(V1 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V2 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V3 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V4 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V5 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V6 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V7 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V8 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V9 = factor(rep(NA, 10), levels = c(lab1, lab2)),
                                                        V10 = factor(rep(NA, 10), levels = c(lab1, lab2))),
                 
                 auc.less.none = matrix(NA, nfolds, nsamples),
                 auc.less.std = matrix(NA, nfolds, nsamples),
                 auc.less.less = matrix(NA, nfolds, nsamples),
                 auc.less.lessstd = matrix(NA, nfolds, nsamples),
                 auc.less.lessstd2 = matrix(NA, nfolds, nsamples),
                 auc.svml1.none = matrix(NA, nfolds, nsamples),
                 auc.svml1.std = matrix(NA, nfolds, nsamples),
                 auc.svml1.less = matrix(NA, nfolds, nsamples),
                 auc.svml1.lessstd = matrix(NA, nfolds, nsamples),
                 auc.svml1.lessstd2 = matrix(NA, nfolds, nsamples),
                 auc.lrl1.none = matrix(NA, nfolds, nsamples),
                 auc.lrl1.std = matrix(NA, nfolds, nsamples),
                 auc.lrl1.less = matrix(NA, nfolds, nsamples),
                 auc.lrl1.lessstd = matrix(NA, nfolds, nsamples),
                 auc.lrl1.lessstd2 = matrix(NA, nfolds, nsamples),
                 auc.lasso.none = matrix(NA, nfolds, nsamples),
                 auc.lasso.std = matrix(NA, nfolds, nsamples),
                 auc.lasso.less = matrix(NA, nfolds, nsamples),
                 auc.lasso.lessstd = matrix(NA, nfolds, nsamples),
                 auc.lasso.lessstd2 = matrix(NA, nfolds, nsamples),
                 
                 accuracy.less.none = matrix(NA, nfolds, nsamples),
                 accuracy.less.std = matrix(NA, nfolds, nsamples),
                 accuracy.less.less = matrix(NA, nfolds, nsamples),
                 accuracy.less.lessstd = matrix(NA, nfolds, nsamples),
                 accuracy.less.lessstd2 = matrix(NA, nfolds, nsamples),
                 accuracy.svml1.none = matrix(NA, nfolds, nsamples),
                 accuracy.svml1.std = matrix(NA, nfolds, nsamples),
                 accuracy.svml1.less = matrix(NA, nfolds, nsamples),
                 accuracy.svml1.lessstd = matrix(NA, nfolds, nsamples),
                 accuracy.svml1.lessstd2 = matrix(NA, nfolds, nsamples),
                 accuracy.lrl1.none = matrix(NA, nfolds, nsamples),
                 accuracy.lrl1.std = matrix(NA, nfolds, nsamples),
                 accuracy.lrl1.less = matrix(NA, nfolds, nsamples),
                 accuracy.lrl1.lessstd = matrix(NA, nfolds, nsamples),
                 accuracy.lrl1.lessstd2 = matrix(NA, nfolds, nsamples),
                 accuracy.lasso.none = matrix(NA, nfolds, nsamples),
                 accuracy.lasso.std = matrix(NA, nfolds, nsamples),
                 accuracy.lasso.less = matrix(NA, nfolds, nsamples),
                 accuracy.lasso.lessstd = matrix(NA, nfolds, nsamples),
                 accuracy.lasso.lessstd2 = matrix(NA, nfolds, nsamples),
                 
                 generror.less.none = matrix(NA, nfolds, nsamples),
                 generror.less.std = matrix(NA, nfolds, nsamples),
                 generror.less.less = matrix(NA, nfolds, nsamples),
                 generror.less.lessstd = matrix(NA, nfolds, nsamples),
                 generror.less.lessstd2 = matrix(NA, nfolds, nsamples),
                 generror.svml1.none = matrix(NA, nfolds, nsamples),
                 generror.svml1.std = matrix(NA, nfolds, nsamples),
                 generror.svml1.less = matrix(NA, nfolds, nsamples),
                 generror.svml1.lessstd = matrix(NA, nfolds, nsamples),
                 generror.svml1.lessstd2 = matrix(NA, nfolds, nsamples),
                 generror.lrl1.none = matrix(NA, nfolds, nsamples),
                 generror.lrl1.std = matrix(NA, nfolds, nsamples),
                 generror.lrl1.less = matrix(NA, nfolds, nsamples),
                 generror.lrl1.lessstd = matrix(NA, nfolds, nsamples),
                 generror.lrl1.lessstd2 = matrix(NA, nfolds, nsamples),
                 generror.lasso.none = matrix(NA, nfolds, nsamples),
                 generror.lasso.std = matrix(NA, nfolds, nsamples),
                 generror.lasso.less = matrix(NA, nfolds, nsamples),
                 generror.lasso.lessstd = matrix(NA, nfolds, nsamples),
                 generror.lasso.lessstd2 = matrix(NA, nfolds, nsamples))

# Go parallel
registerDoParallel(cores = ncores)

sim.list <- foreach(k = 1:K) %dopar% {
  
  print(paste("Repetition", k, "/", K))
  
  set.seed(0952702 + k)
  ncores <- 50   # number of parallel cpu cores
  K <- 100       # number of repetitions to average over
  nfolds <- 10   # number of folds for results
  nfolds.cv <- 4 # number of folds for cross-validation for hyper parameter
  nsamples <- 10 # number of different sample sizes to test
  
  # Load packages
  library(foreach)
  library(parallel)
  library(iterators)
  library(doParallel)
  library(glmnet)
  library(lpSolve)
  library(pracma)
  library(ggplot2)
  library(pROC)
  library(R.matlab)
  library(LiblineaR)
  library(telegram)
  library(latex2exp)
  library(scales)
  source("Scripts/lesstwoc_mv.R")
  source("Scripts/Telegram.R")  
  
  # Load data
  mydata <- readMat("Data/glioma.mat")
  mydata <- cbind(as.data.frame(mydata$a$data), y = factor(mydata$a$nlab))
  dim(mydata) # 50, 4435
  table(mydata[, ncol(mydata)]) # 14, 7, 14, 15
  
  # Response variable contains 4 categories >>> 1+3 vs 2+4
  mydata[, ncol(mydata)] <- ifelse(mydata[, ncol(mydata)] == 1 | mydata[, ncol(mydata)] == 3, 0, 1)
  table(mydata[, ncol(mydata)]) # 28, 22
  mydata[, ncol(mydata)] <- as.factor(mydata[, ncol(mydata)])
  lab1 <- levels(mydata[, ncol(mydata)])[1] # "0"
  lab2 <- levels(mydata[, ncol(mydata)])[2] # "1"
  row.names(mydata) <- 1:nrow(mydata)
  
  # Cross-validation settings
  # Specify folds for class 1
  c1 <- as.numeric(row.names(mydata[mydata[, ncol(mydata)] == lab1, ]))
  fold.id.c1 <- sample(rep(seq(nfolds), length = length(c1)))
  
  # Specify folds for class 2
  c2 <- as.numeric(row.names(mydata[mydata[, ncol(mydata)] == lab2, ]))
  fold.id.c2 <- sample(rep(seq(nfolds), length = length(c2)))
  
  # 10 fold cross-validation (outer loop)
  for (fold in 1:nfolds) {
    print(paste("Fold", fold))
    train <- rbind(mydata[c1[fold.id.c1 != fold], ],
                   mydata[c2[fold.id.c2 != fold], ])
    
    test <- rbind(mydata[c1[fold.id.c1 == fold], ],
                  mydata[c2[fold.id.c2 == fold], ])
    
    # indices per class
    c1.2 <- as.numeric(row.names(train[train[, ncol(train)] == lab1, ]))
    c2.2 <- as.numeric(row.names(train[train[, ncol(train)] == lab2, ]))

    ## Sample n observations per class from train set from specified fold
    for (n in 1:nsamples) {
      index.n <- c(sample(c1.2, n1[n]), sample(c2.2, n2[n]))
      train.n <- train[as.numeric(row.names(train)) %in% index.n, ]
      
      # fold id per class (inner cv loop)
      foldid.c1.2 <- sample(rep(seq(nfolds.cv), length = n1[n]))
      foldid.c2.2 <- sample(rep(seq(nfolds.cv), length = n2[n]))
      foldid.cv <- c(foldid.c1.2, foldid.c2.2)
      
      ## Data scaling based on train.n set
      
      # Map data for standardisation
      train.n.std <- train.n
      for (j in 2:ncol(train.n) - 1) {
        train.n.std[, j] <- (train.n[, j] - mean(train.n[, j])) / sd(train.n[, j]) ^ as.logical(sd(train.n[, j]))
      }
      
      test.std <- test
      for (j in 2:ncol(train.n) - 1) {
        test.std[, j] <- (test[, j] - mean(train.n[, j])) / sd(train.n[, j]) ^ as.logical(sd(train.n[, j]))
      }
      
      # Map data for LESS scaling
      M.train.n <- matrix(0.0, nrow = length(levels(train.n[, ncol(train.n)])), ncol = ncol(train.n) - 1)
      M.train.n[1, ] <- colMeans(train.n[train.n[, ncol(train.n)] == lab1, -ncol(train.n)])
      M.train.n[2, ] <- colMeans(train.n[train.n[, ncol(train.n)] == lab2, -ncol(train.n)])
      train.n.map <- train.n
      train.n.map[, -ncol(train.n.map)] <- mapmeans(DF = train.n[, -ncol(train.n)], M = M.train.n)
      
      test.map <- test
      test.map[, -ncol(test.map)] <- mapmeans(DF = test[, -ncol(test)], M = M.train.n)

      # Map data for LESSstd scaling
      S.train.n <- matrix(0.0, nrow = length(levels(train.n[, ncol(train.n)])), ncol = ncol(train.n) - 1)
      S.train.n[1, ] <- apply(train.n[train.n[, ncol(train.n)] == lab1, -ncol(train.n)], 2, var)
      S.train.n[2, ] <- apply(train.n[train.n[, ncol(train.n)] == lab2, -ncol(train.n)], 2, var)
      S.train.n <- ifelse(S.train.n == 0, unique(sort(S.train.n))[2], S.train.n) # When var = 0
      
      train.n.map.std <- train.n
      train.n.map.std[, -ncol(train.n.map.std)] <- mapmeansstd(DF = train.n[, -ncol(train.n)], 
                                                               M = M.train.n, 
                                                               S = S.train.n)

      test.map.std <- test
      test.map.std[, -ncol(test.map.std)] <- mapmeansstd(DF = test[, -ncol(test)], 
                                                         M = M.train.n, 
                                                         S = S.train.n)
      
      # Map data for LESSstd2 scaling
      S.train.n2 <- matrix(apply(rbind(train.n[train.n[, ncol(train.n)] == lab1, -ncol(train.n)] -
                                         matrix(rep(M.train.n[1, ],
                                                    times = sum(train.n[, ncol(train.n)] == lab1)),
                                              nrow = sum(train.n[, ncol(train.n)] == lab1),
                                              byrow = TRUE),
                                       train.n[train.n[, ncol(train.n)] == lab2, -ncol(train.n)] -
                                         matrix(rep(M.train.n[2, ],
                                                    times = sum(train.n[, ncol(train.n)] == lab2)),
                                              nrow = sum(train.n[, ncol(train.n)] == lab2),
                                              byrow = TRUE)),
                                 2, var),
                           nrow = 1, ncol = ncol(train.n) - 1)
      S.train.n2 <- ifelse(S.train.n2 == 0, unique(sort(S.train.n2))[2], S.train.n2) # When var = 0
      
      train.n.map.std2 <- train.n
      train.n.map.std2[, -ncol(train.n)] <- mapmeansstd2(DF = train.n[, -ncol(train.n)], 
                                                         M = M.train.n, 
                                                         S = S.train.n2)
      
      test.map.std2 <- test
      test.map.std2[, -ncol(test)] <- mapmeansstd2(DF = test[, -ncol(test)], M = M.train.n, S = S.train.n2)
      
      
      #### LESS ################################################################
      
      #### LESS + no scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.less.none <- 10^seq(-4, 1, length.out = 11)
      # C.heuristic.less.none <- heuristicC(data = as.matrix(train.n[, -ncol(train.n)]))
      C.init.less.none <- 0.03162278 # based on cv on whole dataset
      C.hyper.less.none <- C.init.less.none * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.less.none <- factor(numeric(nrow(train.n)), levels = c(lab1, lab2))
      error.cv.less.none <- numeric(length(C.hyper.less.none))
      # for (c in C.hyper.less.none) {
      #   for (loocv in 1:nrow(train.n)) {
      #     model.loocv.less.none <- lesstwoc_none(DF = train.n[-loocv, ],
      #                                            C = c)
      #     pred.loocv.less.none[loocv] <- predict.less_none(MODEL = model.loocv.less.none,
      #                                                      NEWDATA = train.n[loocv, -ncol(train.n)])$prediction
      #   }
      #   error.loocv.less.none[which(c == C.hyper.less.none)] <- mean(pred.loocv.less.none != train.n[, ncol(train.n)])
      # }
      for (c in C.hyper.less.none) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.less.none <- lesstwoc_none(DF = train.n[foldid.cv != fold.cv, ],
                                              C = c)
          pred.cv.less.none[foldid.cv == fold.cv] <- predict.less_none(MODEL = model.cv.less.none,
                                                                       NEWDATA = train.n[foldid.cv == fold.cv, -ncol(train.n)])$prediction
        }
        error.cv.less.none[which(c == C.hyper.less.none)] <- mean(pred.cv.less.none != train.n[, ncol(train.n)])
      }
      sim.list$C.cv.less.none[fold, n] <- C.hyper.less.none[
        which(error.cv.less.none == min(error.cv.less.none))[
          ceiling(median(1:length(which(error.cv.less.none == min(error.cv.less.none)))))]]
      
      # Train model
      model.less.none <- lesstwoc_none(DF = train.n, C = sim.list$C.cv.less.none[fold, n])
      sim.list$numbetas.less.none[fold, n] <- sum(model.less.none$model$beta != 0)
      
      # Test model
      sim.list$preds.less.none[as.numeric(row.names(test)), n] <- predict.less_none(MODEL = model.less.none, NEWDATA = test[, -ncol(test)])$score
      sim.list$preds.labs.less.none[as.numeric(row.names(test)), n] <- factor(predict.less_none(MODEL = model.less.none, NEWDATA = test[, -ncol(test)])$prediction, levels = c(lab1, lab2))
      
      
      #### LESS + standardisation ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.less.std <- 10^seq(-4, 1, length.out = 11)
      # C.heuristic.less.std <- heuristicC(data = as.matrix(train.n[, -ncol(train.n)]))
      C.init.less.std <- 0.06309573 # based on cv on whole dataset
      C.hyper.less.std <- C.init.less.std * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.less.std <- factor(numeric(nrow(train.n)), levels = c(lab1, lab2))
      error.cv.less.std <- numeric(length(C.hyper.less.std))
      for (c in C.hyper.less.std) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.less.std <- lesstwoc_std(DF = train.n[foldid.cv != fold.cv, ],
                                            C = c)
          # LOOCV can't standardize based on 1 sample, therefore do manual and use predict.less_none
          pred.cv.less.std[foldid.cv == fold.cv] <- predict.less_std(MODEL = model.cv.less.std,
                                                                     NEWDATA = train.n[foldid.cv == fold.cv, -ncol(train.n)])$prediction
        }
        error.cv.less.std[which(c == C.hyper.less.std)] <- mean(pred.cv.less.std != train.n[, ncol(train.n)])
      }
      sim.list$C.cv.less.std[fold, n] <- C.hyper.less.std[
        which(error.cv.less.std == min(error.cv.less.std))[
          ceiling(median(1:length(which(error.cv.less.std == min(error.cv.less.std)))))]]
      
      # Train model
      model.less.std <- lesstwoc_std(DF = train.n, C = sim.list$C.cv.less.std[fold, n])
      sim.list$numbetas.less.std[fold, n] <- sum(model.less.std$model$beta != 0)
      
      # Test model
      sim.list$preds.less.std[as.numeric(row.names(test)), n] <- predict.less_std(MODEL = model.less.std, NEWDATA = test[, -ncol(test)])$score
      sim.list$preds.labs.less.std[as.numeric(row.names(test)), n] <- factor(predict.less_std(MODEL = model.less.std, NEWDATA = test[, -ncol(test)])$prediction, levels = c(lab1, lab2))
      
      
      #### LESS + LESS scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.less.less <- 10^seq(-4, 1, length.out = 11)
      # C.heuristic.less.less <- heuristicC(data = as.matrix(train.n[, -ncol(train.n)]))
      C.init.less.less <- 0.4570882 # based on cv on whole dataset
      C.hyper.less.less <- C.init.less.less * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.less.less <- factor(numeric(nrow(train.n)), levels = c(lab1, lab2))
      error.cv.less.less <- numeric(length(C.hyper.less.less))
      for (c in C.hyper.less.less) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.less.less <- lesstwoc(DF = train.n[foldid.cv != fold.cv, ],
                                         C = c)
          pred.cv.less.less[foldid.cv == fold.cv] <- predict.less(MODEL = model.cv.less.less,
                                                      NEWDATA = train.n[foldid.cv == fold.cv, -ncol(train.n)])$prediction
        }
        error.cv.less.less[which(c == C.hyper.less.less)] <- mean(pred.cv.less.less != train.n[, ncol(train.n)])
      }
      sim.list$C.cv.less.less[fold, n] <- C.hyper.less.less[
        which(error.cv.less.less == min(error.cv.less.less))[
          ceiling(median(1:length(which(error.cv.less.less == min(error.cv.less.less)))))]]
      
      # Train model
      model.less.less <- lesstwoc(DF = train.n, C = sim.list$C.cv.less.less[fold, n])
      sim.list$numbetas.less.less[fold, n] <- sum(model.less.less$model$beta != 0)
      
      # Test model
      sim.list$preds.less.less[as.numeric(row.names(test)), n] <- predict.less(MODEL = model.less.less, NEWDATA = test[, -ncol(test)])$score
      sim.list$preds.labs.less.less[as.numeric(row.names(test)), n] <- factor(predict.less(MODEL = model.less.less, NEWDATA = test[, -ncol(test)])$prediction, levels = c(lab1, lab2))
      
      
      #### LESS + LESSstd scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.less.lessstd <- 10^seq(-4, 1, length.out = 11)
      # C.heuristic.less.lessstd <- heuristicC(data = as.matrix(train.n[, -ncol(train.n)]))
      C.init.less.lessstd <- 0.006606934 # based on cv on whole dataset
      C.hyper.less.lessstd <- C.init.less.lessstd * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.less.lessstd <- factor(numeric(nrow(train.n)), levels = c(lab1, lab2))
      error.cv.less.lessstd <- numeric(length(C.hyper.less.lessstd))
      for (c in C.hyper.less.lessstd) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.less.lessstd <- lesstwoc_lessstd(DF = train.n[foldid.cv != fold.cv, ],
                                                    C = c)
          pred.cv.less.lessstd[foldid.cv == fold.cv] <- predict.less_lessstd(MODEL = model.cv.less.lessstd,
                                                           NEWDATA = train.n[foldid.cv == fold.cv, -ncol(train.n)])$prediction
        }
        error.cv.less.lessstd[which(c == C.hyper.less.lessstd)] <- mean(pred.cv.less.lessstd != train.n[, ncol(train.n)])
      }
      sim.list$C.cv.less.lessstd[fold, n] <- C.hyper.less.lessstd[
        which(error.cv.less.lessstd == min(error.cv.less.lessstd))[
          ceiling(median(1:length(which(error.cv.less.lessstd == min(error.cv.less.lessstd)))))]]
      
      # Train model
      model.less.lessstd <- lesstwoc_lessstd(DF = train.n, C = sim.list$C.cv.less.lessstd[fold, n])
      sim.list$numbetas.less.lessstd[fold, n] <- sum(model.less.lessstd$model$beta != 0)
      
      # Test model
      sim.list$preds.less.lessstd[as.numeric(row.names(test)), n] <- predict.less_lessstd(MODEL = model.less.lessstd, NEWDATA = test[, -ncol(test)])$score
      sim.list$preds.labs.less.lessstd[as.numeric(row.names(test)), n] <- factor(predict.less_lessstd(MODEL = model.less.lessstd, NEWDATA = test[, -ncol(test)])$prediction, levels = c(lab1, lab2))
      
      
      #### LESS + LESSstd2 scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.less.lessstd2 <- 10^seq(-4, 1, length.out = 11)
      # C.heuristic.less.lessstd2 <- heuristicC(data = as.matrix(train.n[, -ncol(train.n)]))
      C.init.less.lessstd2 <- 0.05495409 # based on cv on whole dataset
      C.hyper.less.lessstd2 <- C.init.less.lessstd2 * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.less.lessstd2 <- factor(numeric(nrow(train.n)), levels = c(lab1, lab2))
      error.cv.less.lessstd2 <- numeric(length(C.hyper.less.lessstd2))
      for (c in C.hyper.less.lessstd2) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.less.lessstd2 <- lesstwoc_lessstd2(DF = train.n[foldid.cv != fold.cv, ],
                                                      C = c)
          pred.cv.less.lessstd2[foldid.cv == fold.cv] <- predict.less_lessstd2(MODEL = model.cv.less.lessstd2,
                                                           NEWDATA = train.n[foldid.cv == fold.cv, -ncol(train.n)])$prediction
        }
        error.cv.less.lessstd2[which(c == C.hyper.less.lessstd2)] <- mean(pred.cv.less.lessstd2 != train.n[, ncol(train.n)])
      }
      sim.list$C.cv.less.lessstd2[fold, n] <- C.hyper.less.lessstd2[
        which(error.cv.less.lessstd2 == min(error.cv.less.lessstd2))[
          ceiling(median(1:length(which(error.cv.less.lessstd2 == min(error.cv.less.lessstd2)))))]]
      
      # Train model
      model.less.lessstd2 <- lesstwoc_lessstd2(DF = train.n, C = sim.list$C.cv.less.lessstd2[fold, n])
      sim.list$numbetas.less.lessstd2[fold, n] <- sum(model.less.lessstd2$model$beta != 0)
      
      # Test model
      sim.list$preds.less.lessstd2[as.numeric(row.names(test)), n] <- predict.less_lessstd2(MODEL = model.less.lessstd2, NEWDATA = test[, -ncol(test)])$score
      sim.list$preds.labs.less.lessstd2[as.numeric(row.names(test)), n] <- factor(predict.less_lessstd2(MODEL = model.less.lessstd2, NEWDATA = test[, -ncol(test)])$prediction, levels = c(lab1, lab2))
      
      
      #### Support Vector Machine with L1 regularisation #######################
      
      #### SVML1 + no scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.svml1.none <- 10^seq(-4, 1, length.out = 11)
      C.init.svml1.none <- 0.2290868 # based on cv on whole dataset
      C.hyper.svml1.none <- C.init.svml1.none * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.svml1.none <- factor(numeric(nrow(train.n)), levels = c(lab1, lab2))
      error.cv.svml1.none <- numeric(length(C.hyper.svml1.none))
      for (c in C.hyper.svml1.none) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.svml1.none <- LiblineaR(data = train.n[foldid.cv != fold.cv, -ncol(train.n)],
                                           target = train.n[foldid.cv != fold.cv, ncol(train.n)],
                                           type = 5, #  L1-regularized L2-loss support vector classification
                                           cost = c, 
                                           epsilon = 1e-7, 
                                           bias = 1,
                                           wi = NULL, 
                                           cross = 0, 
                                           verbose = FALSE,
                                           findC = FALSE, 
                                           useInitC = FALSE)
          pred.cv.svml1.none[foldid.cv == fold.cv] <- predict(model.cv.svml1.none,
                                                  train.n[foldid.cv == fold.cv, -ncol(train.n)],
                                                  decisionValues = FALSE)$predictions
        }
        error.cv.svml1.none[which(c == C.hyper.svml1.none)] <- mean(pred.cv.svml1.none != train.n[, ncol(train.n)])
      }
      sim.list$C.cv.svml1.none[fold, n] <- C.hyper.svml1.none[
        which(error.cv.svml1.none == min(error.cv.svml1.none))[
          ceiling(median(1:length(which(error.cv.svml1.none == min(error.cv.svml1.none)))))]]
      
      # Train model
      model.svml1.none <- LiblineaR(data = as.matrix(train.n[, -ncol(train.n)]),
                                    target = train.n[, ncol(train.n)],
                                    type = 5, #  L1-regularized L2-loss support vector classification
                                    cost = sim.list$C.cv.svml1.none[fold, n], 
                                    epsilon = 1e-7, 
                                    bias = 1,
                                    wi = NULL, 
                                    cross = 0, 
                                    verbose = FALSE,
                                    findC = FALSE, 
                                    useInitC = FALSE)
      sim.list$numbetas.svml1.none[fold, n] <- sum(model.svml1.none$W[-length(model.svml1.none$W)] != 0)
      
      # Test model
      sim.list$preds.svml1.none[as.numeric(row.names(test)), n] <- predict(model.svml1.none, 
                                                                           test[, -ncol(test)],
                                                                           decisionValues = TRUE)$decisionValues[, 1]
      sim.list$preds.labs.svml1.none[as.numeric(row.names(test)), n] <- factor(predict(model.svml1.none,
                                                                                       test[, -ncol(test)])$predictions,
                                                                               levels = c(lab1, lab2))
      
      
      #### SVML1 + standardisation ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.svml1.std <- 10^seq(-4, 1, length.out = 11)
      C.init.svml1.std <- 0.03630781 # based on cv on whole dataset
      C.hyper.svml1.std <- C.init.svml1.std * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.svml1.std <- factor(numeric(nrow(train.n)), levels = c(lab1, lab2))
      error.cv.svml1.std <- numeric(length(C.hyper.svml1.std))
      for (c in C.hyper.svml1.std) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.svml1.std <- LiblineaR(data = train.n.std[foldid.cv != fold.cv, -ncol(train.n.std)],
                                          target = train.n.std[foldid.cv != fold.cv, ncol(train.n.std)],
                                          type = 5, #  L1-regularized L2-loss support vector classification
                                          cost = c, 
                                          epsilon = 1e-7, 
                                          bias = 1,
                                          wi = NULL, 
                                          cross = 0, 
                                          verbose = FALSE,
                                          findC = FALSE, 
                                          useInitC = FALSE)
          pred.cv.svml1.std[foldid.cv == fold.cv] <- predict(model.cv.svml1.std,
                                                             train.n.std[foldid.cv == fold.cv, -ncol(train.n.std)],
                                                             decisionValues = FALSE)$predictions
        }
        error.cv.svml1.std[which(c == C.hyper.svml1.std)] <- mean(pred.cv.svml1.std != train.n.std[, ncol(train.n.std)])
      }
      sim.list$C.cv.svml1.std[fold, n] <- C.hyper.svml1.std[
        which(error.cv.svml1.std == min(error.cv.svml1.std))[
          ceiling(median(1:length(which(error.cv.svml1.std == min(error.cv.svml1.std)))))]]
      
      # Train model
      model.svml1.std <- LiblineaR(data = as.matrix(train.n.std[, -ncol(train.n.std)]),
                                    target = train.n.std[, ncol(train.n.std)],
                                    type = 5, #  L1-regularized L2-loss support vector classification
                                    cost = sim.list$C.cv.svml1.std[fold, n], 
                                    epsilon = 1e-7, 
                                    bias = 1,
                                    wi = NULL, 
                                    cross = 0, 
                                    verbose = FALSE,
                                    findC = FALSE, 
                                    useInitC = FALSE)
      sim.list$numbetas.svml1.std[fold, n] <- sum(model.svml1.std$W[-length(model.svml1.std$W)] != 0)
      
      # Test model
      sim.list$preds.svml1.std[as.numeric(row.names(test)), n] <- predict(model.svml1.std, 
                                                                          test.std[, -ncol(test.std)],
                                                                          decisionValues = TRUE)$decisionValues[, 1]
      sim.list$preds.labs.svml1.std[as.numeric(row.names(test)), n] <- factor(predict(model.svml1.std,
                                                                                      test.std[, -ncol(test.std)])$predictions,
                                                                               levels = c(lab1, lab2))
      
      
      #### SVML1 + LESS scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.svml1.less <- 10^seq(-4, 1, length.out = 11)
      C.init.svml1.less <- 0.5495409 # based on cv on whole dataset
      C.hyper.svml1.less <- C.init.svml1.less * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.svml1.less <- factor(numeric(nrow(train.n.map)), levels = c(lab1, lab2))
      error.cv.svml1.less <- numeric(length(C.hyper.svml1.less))
      for (c in C.hyper.svml1.less) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.svml1.less <- LiblineaR(data = train.n.map[foldid.cv != fold.cv, -ncol(train.n.map)],
                                           target = train.n.map[foldid.cv != fold.cv, ncol(train.n.map)],
                                           type = 5, #  L1-regularized L2-loss support vector classification
                                           cost = c, 
                                           epsilon = 1e-7, 
                                           bias = 1,
                                           wi = NULL, 
                                           cross = 0, 
                                           verbose = FALSE,
                                           findC = FALSE, 
                                           useInitC = FALSE)
          pred.cv.svml1.less[foldid.cv == fold.cv] <- predict(model.cv.svml1.less,
                                                              train.n.map[foldid.cv == fold.cv, -ncol(train.n.map)],
                                                              decisionValues = FALSE)$predictions
        }
        error.cv.svml1.less[which(c == C.hyper.svml1.less)] <- mean(pred.cv.svml1.less != train.n.map[, ncol(train.n.map)])
      }
      sim.list$C.cv.svml1.less[fold, n] <- C.hyper.svml1.less[
        which(error.cv.svml1.less == min(error.cv.svml1.less))[
          ceiling(median(1:length(which(error.cv.svml1.less == min(error.cv.svml1.less)))))]]
      
      # Train model
      model.svml1.less <- LiblineaR(data = as.matrix(train.n.map[, -ncol(train.n.map)]),
                                    target = train.n.map[, ncol(train.n.map)],
                                    type = 5, #  L1-regularized L2-loss support vector classification
                                    cost = sim.list$C.cv.svml1.less[fold, n], 
                                    epsilon = 1e-7, 
                                    bias = 1,
                                    wi = NULL, 
                                    cross = 0, 
                                    verbose = FALSE,
                                    findC = FALSE, 
                                    useInitC = FALSE)
      sim.list$numbetas.svml1.less[fold, n] <- sum(model.svml1.less$W[-length(model.svml1.less$W)] != 0)
      
      # Test model
      sim.list$preds.svml1.less[as.numeric(row.names(test.map)), n] <- predict(model.svml1.less, 
                                                                               test.map[, -ncol(test.map)],
                                                                               decisionValues = TRUE)$decisionValues[, 1]
      sim.list$preds.labs.svml1.less[as.numeric(row.names(test.map)), n] <- factor(predict(model.svml1.less,
                                                                                           test.map[, -ncol(test.map)])$predictions,
                                                                                   levels = c(lab1, lab2))
      
      
      #### SVML1 + LESSstd scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.svml1.lessstd <- 10^seq(-4, 1, length.out = 11)
      C.init.svml1.lessstd <- 0.02290868 # based on cv on whole dataset
      C.hyper.svml1.lessstd <- C.init.svml1.lessstd * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.svml1.lessstd <- factor(numeric(nrow(train.n.map.std)), levels = c(lab1, lab2))
      error.cv.svml1.lessstd <- numeric(length(C.hyper.svml1.lessstd))
      for (c in C.hyper.svml1.lessstd) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.svml1.lessstd <- LiblineaR(data = train.n.map.std[foldid.cv != fold.cv, -ncol(train.n.map.std)],
                                              target = train.n.map.std[foldid.cv != fold.cv, ncol(train.n.map.std)],
                                              type = 5, #  L1-regularized L2-loss support vector classification
                                              cost = c, 
                                              epsilon = 1e-7, 
                                              bias = 1,
                                              wi = NULL, 
                                              cross = 0, 
                                              verbose = FALSE,
                                              findC = FALSE, 
                                              useInitC = FALSE)
          pred.cv.svml1.lessstd[foldid.cv == fold.cv] <- predict(model.cv.svml1.lessstd,
                                                     train.n.map.std[foldid.cv == fold.cv, -ncol(train.n.map.std)],
                                                     decisionValues = FALSE)$predictions
        }
        error.cv.svml1.lessstd[which(c == C.hyper.svml1.lessstd)] <- mean(pred.cv.svml1.lessstd != train.n.map.std[, ncol(train.n.map.std)])
      }
      sim.list$C.cv.svml1.lessstd[fold, n] <- C.hyper.svml1.lessstd[
        which(error.cv.svml1.lessstd == min(error.cv.svml1.lessstd))[
          ceiling(median(1:length(which(error.cv.svml1.lessstd == min(error.cv.svml1.lessstd)))))]]
      
      # Train model
      model.svml1.lessstd <- LiblineaR(data = as.matrix(train.n.map.std[, -ncol(train.n.map.std)]),
                                       target = train.n.map.std[, ncol(train.n.map.std)],
                                       type = 5, #  L1-regularized L2-loss support vector classification
                                       cost = sim.list$C.cv.svml1.lessstd[fold, n], 
                                       epsilon = 1e-7, 
                                       bias = 1,
                                       wi = NULL, 
                                       cross = 0, 
                                       verbose = FALSE,
                                       findC = FALSE, 
                                       useInitC = FALSE)
      sim.list$numbetas.svml1.lessstd[fold, n] <- sum(model.svml1.lessstd$W[-length(model.svml1.lessstd$W)] != 0)
      
      # Test model
      sim.list$preds.svml1.lessstd[as.numeric(row.names(test.map.std)), n] <- predict(model.svml1.lessstd, 
                                                                           test.map.std[, -ncol(test.map.std)],
                                                                           decisionValues = TRUE)$decisionValues[, 1]
      sim.list$preds.labs.svml1.lessstd[as.numeric(row.names(test.map.std)), n] <- factor(predict(model.svml1.lessstd,
                                                                                       test.map.std[, -ncol(test.map.std)])$predictions,
                                                                               levels = c(lab1, lab2))
      
      
      #### SVML1 + LESSstd2 scaling ####
      
      # Leave-one-out cross-validation for penalisation parameter C
      # C.hyper.svml1.lessstd2 <- 10^seq(-4, 1, length.out = 11)
      C.init.svml1.lessstd2 <- 3.981072 # based on cv on whole dataset
      C.hyper.svml1.lessstd2 <- C.init.svml1.lessstd2 * 10^seq(-0.5, 0.5, length.out = 11)
      pred.cv.svml1.lessstd2 <- factor(numeric(nrow(train.n.map.std2)), levels = c(lab1, lab2))
      error.cv.svml1.lessstd2 <- numeric(length(C.hyper.svml1.lessstd2))
      for (c in C.hyper.svml1.lessstd2) {
        for (fold.cv in 1:nfolds.cv) {
          model.cv.svml1.lessstd2 <- LiblineaR(data = train.n.map.std2[foldid.cv != fold.cv, -ncol(train.n.map.std2)],
                                               target = train.n.map.std2[foldid.cv != fold.cv, ncol(train.n.map.std2)],
                                               type = 5, #  L1-regularized L2-loss support vector classification
                                               cost = c, 
                                               epsilon = 1e-7, 
                                               bias = 1,
                                               wi = NULL, 
                                               cross = 0, 
                                               verbose = FALSE,
                                               findC = FALSE, 
                                               useInitC = FALSE)
          pred.cv.svml1.lessstd2[foldid.cv == fold.cv] <- predict(model.cv.svml1.lessstd2,
                                                     train.n.map.std2[foldid.cv == fold.cv, -ncol(train.n.map.std2)],
                                                     decisionValues = FALSE)$predictions
        }
        error.cv.svml1.lessstd2[which(c == C.hyper.svml1.lessstd2)] <- mean(pred.cv.svml1.lessstd2 != train.n.map.std2[, ncol(train.n.map.std2)])
      }
      sim.list$C.cv.svml1.lessstd2[fold, n] <- C.hyper.svml1.lessstd2[
        which(error.cv.svml1.lessstd2 == min(error.cv.svml1.lessstd2))[
          ceiling(median(1:length(which(error.cv.svml1.lessstd2 == min(error.cv.svml1.lessstd2)))))]]
      
      # Train model
      model.svml1.lessstd2 <- LiblineaR(data = as.matrix(train.n.map.std2[, -ncol(train.n.map.std2)]),
                                        target = train.n.map.std2[, ncol(train.n.map.std2)],
                                        type = 5, #  L1-regularized L2-loss support vector classification
                                        cost = sim.list$C.cv.svml1.lessstd2[fold, n], 
                                        epsilon = 1e-7, 
                                        bias = 1,
                                        wi = NULL, 
                                        cross = 0, 
                                        verbose = FALSE,
                                        findC = FALSE, 
                                        useInitC = FALSE)
      sim.list$numbetas.svml1.lessstd2[fold, n] <- sum(model.svml1.lessstd2$W[-length(model.svml1.lessstd2$W)] != 0)
      
      # Test model
      sim.list$preds.svml1.lessstd2[as.numeric(row.names(test.map.std2)), n] <- predict(model.svml1.lessstd2, 
                                                                                      test.map.std2[, -ncol(test.map.std2)],
                                                                                      decisionValues = TRUE)$decisionValues[, 1]
      sim.list$preds.labs.svml1.lessstd2[as.numeric(row.names(test.map.std2)), n] <- factor(predict(model.svml1.lessstd2,
                                                                                                  test.map.std2[, -ncol(test.map.std2)])$predictions,
                                                                                          levels = c(lab1, lab2))
      
      
      #### Logistic Regression with L1 penalisation ############################
      
      #### LRL1 + no scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lrl1.none <- cv.glmnet(x = as.matrix(train.n[, -ncol(train.n)]),
                                      y = train.n[, ncol(train.n)],
                                      family = "binomial",
                                      alpha = 1,
                                      foldid = foldid.cv,
                                      type.measure = "class",
                                      grouped = FALSE)
      
      # Train model
      model.lrl1.none <- glmnet(x = as.matrix(train.n[, -ncol(train.n)]), 
                                y = train.n[, ncol(train.n)],
                                intercept = TRUE, 
                                standardize = FALSE,
                                family = "binomial",
                                alpha = 1, 
                                lambda = cv.model.lrl1.none$lambda.1se)
      sim.list$numbetas.lrl1.none[fold, n] <- sum(coef(model.lrl1.none)[-1] != 0)
      
      # Test model
      sim.list$preds.lrl1.none[as.numeric(row.names(test)), n] <- predict.glmnet(object = model.lrl1.none,
                                                                                 newx = as.matrix(test[, -ncol(test)]),
                                                                                 s = cv.model.lrl1.none$lambda.1se, 
                                                                                 type = "class")
      
      
      #### LRL1 + standardisation ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lrl1.std <- cv.glmnet(x = as.matrix(train.n.std[, -ncol(train.n.std)]),
                                     y = train.n.std[, ncol(train.n.std)],
                                     family = "binomial",
                                     alpha = 1,
                                     foldid = foldid.cv,
                                     type.measure = "class",
                                     grouped = FALSE)
      
      # Train model
      model.lrl1.std <- glmnet(x = as.matrix(train.n.std[, -ncol(train.n.std)]),
                               y = train.n.std[, ncol(train.n.std)],
                               intercept = TRUE, 
                               standardize = FALSE,
                               family = "binomial",
                               alpha = 1, 
                               lambda = cv.model.lrl1.std$lambda.1se)
      sim.list$numbetas.lrl1.std[fold, n] <- sum(coef(model.lrl1.std)[-1] != 0)
      
      # Test model
      sim.list$preds.lrl1.std[as.numeric(row.names(test)), n] <- predict.glmnet(object = model.lrl1.std,
                                                                                newx = as.matrix(test.std[, -ncol(test.std)]),
                                                                                s = cv.model.lrl1.std$lambda.1se, 
                                                                                type = "class")
      
      
      #### LRL1 + LESS scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lrl1.less <- cv.glmnet(x = as.matrix(train.n.map[, -ncol(train.n.map)]),
                                      y = train.n.map[, ncol(train.n.map)],
                                      family = "binomial",
                                      alpha = 1,
                                      foldid = foldid.cv,
                                      type.measure = "class",
                                      grouped = FALSE)
      
      # Train model
      model.lrl1.less <- glmnet(x = as.matrix(train.n.map[, -ncol(train.n.map)]), 
                                y = train.n.map[, ncol(train.n.map)],
                                intercept = TRUE, 
                                standardize = FALSE,
                                family = "binomial",
                                alpha = 1, 
                                lambda = cv.model.lrl1.less$lambda.1se)
      sim.list$numbetas.lrl1.less[fold, n] <- sum(coef(model.lrl1.less)[-1] != 0)
      
      # Test model
      sim.list$preds.lrl1.less[as.numeric(row.names(test.map)), n] <- predict.glmnet(object = model.lrl1.less,
                                                                                     newx = as.matrix(test.map[, -ncol(test.map)]),
                                                                                     s = cv.model.lrl1.less$lambda.1se, 
                                                                                     type = "class")
      
      
      #### LRL1 + LESSstd scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lrl1.lessstd <- cv.glmnet(x = as.matrix(train.n.map.std[, -ncol(train.n.map.std)]),
                                         y = train.n.map.std[, ncol(train.n.map.std)],
                                         family = "binomial",
                                         alpha = 1,
                                         foldid = foldid.cv,
                                         type.measure = "class",
                                         grouped = FALSE)
      
      # Train model
      model.lrl1.lessstd <- glmnet(x = as.matrix(train.n.map.std[, -ncol(train.n.map.std)]), 
                                y = train.n.map.std[, ncol(train.n.map.std)],
                                intercept = TRUE, 
                                standardize = FALSE,
                                family = "binomial",
                                alpha = 1, 
                                lambda = cv.model.lrl1.lessstd$lambda.1se)
      sim.list$numbetas.lrl1.lessstd[fold, n] <- sum(coef(model.lrl1.lessstd)[-1] != 0)
      
      # Test model
      sim.list$preds.lrl1.lessstd[as.numeric(row.names(test.map.std)), n] <- predict.glmnet(object = model.lrl1.lessstd,
                                                                                 newx = as.matrix(test.map.std[, -ncol(test.map.std)]),
                                                                                 s = cv.model.lrl1.lessstd$lambda.1se, 
                                                                                 type = "class")
      
      
      #### LRL1 + LESSstd2 scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lrl1.lessstd2 <- cv.glmnet(x = as.matrix(train.n.map.std2[, -ncol(train.n.map.std2)]),
                                            y = train.n.map.std2[, ncol(train.n.map.std2)],
                                            family = "binomial",
                                            alpha = 1,
                                            foldid = foldid.cv,
                                            type.measure = "class",
                                            grouped = FALSE)
      
      # Train model
      model.lrl1.lessstd2 <- glmnet(x = as.matrix(train.n.map.std2[, -ncol(train.n.map.std2)]), 
                                   y = train.n.map.std2[, ncol(train.n.map.std2)],
                                   intercept = TRUE, 
                                   standardize = FALSE,
                                   family = "binomial",
                                   alpha = 1, 
                                   lambda = cv.model.lrl1.lessstd2$lambda.1se)
      sim.list$numbetas.lrl1.lessstd2[fold, n] <- sum(coef(model.lrl1.lessstd2)[-1] != 0)
      
      # Test model
      sim.list$preds.lrl1.lessstd2[as.numeric(row.names(test.map.std2)), n] <- predict.glmnet(object = model.lrl1.lessstd2,
                                                                                            newx = as.matrix(test.map.std2[, -ncol(test.map.std2)]),
                                                                                            s = cv.model.lrl1.lessstd2$lambda.1se, 
                                                                                            type = "class")
      
      
      #### LASSO Regression ######################################################
      
      #### LASSO + no scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lasso.none <- cv.glmnet(x = as.matrix(train.n[, -ncol(train.n)]),
                                       y = as.numeric(train.n[, ncol(train.n)]),
                                       family = "gaussian",
                                       alpha = 1,
                                       foldid = foldid.cv,
                                       type.measure = "mse",
                                       grouped = FALSE)
      
      # Train model
      model.lasso.none <- glmnet(x = as.matrix(train.n[, -ncol(train.n)]),
                                 y = as.numeric(train.n[, ncol(train.n)]),
                                 intercept = TRUE,
                                 standardize = FALSE,
                                 family = "gaussian",
                                 alpha = 1,
                                 lambda = cv.model.lasso.none$lambda.1se)
      sim.list$numbetas.lasso.none[fold, n] <- sum(coef(model.lasso.none)[-1] != 0)
      
      # Test model
      sim.list$preds.lasso.none[as.numeric(row.names(test)), n] <- predict.glmnet(object = model.lasso.none,
                                                                                  newx = as.matrix(test[, -ncol(test)]),
                                                                                  s = cv.model.lasso.none$lambda.1se,
                                                                                  type = "response")
      
      
      #### LASSO + standardisation ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lasso.std <- cv.glmnet(x = as.matrix(train.n.std[, -ncol(train.n.std)]),
                                      y = as.numeric(train.n.std[, ncol(train.n.std)]),
                                      family = "gaussian",
                                      alpha = 1,
                                      foldid = foldid.cv,
                                      type.measure = "mse",
                                      grouped = FALSE)
      
      # Train model
      model.lasso.std <- glmnet(x = as.matrix(train.n.std[, -ncol(train.n.std)]),
                                y = as.numeric(train.n[, ncol(train.n)]),
                                intercept = TRUE,
                                standardize = FALSE,
                                family = "gaussian",
                                alpha = 1,
                                lambda = cv.model.lasso.std$lambda.1se)
      sim.list$numbetas.lasso.std[fold, n] <- sum(coef(model.lasso.std)[-1] != 0)
      
      # Test model
      sim.list$preds.lasso.std[as.numeric(row.names(test)), n] <- predict.glmnet(object = model.lasso.std,
                                                                                  newx = as.matrix(test.std[, -ncol(test.std)]),
                                                                                  s = cv.model.lasso.std$lambda.1se,
                                                                                  type = "response")
      
      
      #### LASSO + LESS scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lasso.less <- cv.glmnet(x = as.matrix(train.n.map[, -ncol(train.n.map)]),
                                          y = as.numeric(train.n.map[, ncol(train.n.map)]),
                                          family = "gaussian",
                                          alpha = 1,
                                          foldid = foldid.cv,
                                          type.measure = "mse",
                                          grouped = FALSE)
      
      # Train model
      model.lasso.less <- glmnet(x = as.matrix(train.n.map[, -ncol(train.n.map)]),
                                 y = as.numeric(train.n.map[, ncol(train.n.map)]),
                                 intercept = TRUE,
                                 standardize = FALSE,
                                 family = "gaussian",
                                 alpha = 1,
                                 lambda = cv.model.lasso.less$lambda.1se)
      sim.list$numbetas.lasso.less[fold, n] <- sum(coef(model.lasso.less)[-1] != 0)
      
      # Test model
      sim.list$preds.lasso.less[as.numeric(row.names(test.map)), n] <- predict.glmnet(object = model.lasso.less,
                                                                                  newx = as.matrix(test.map[, -ncol(test.map)]),
                                                                                  s = cv.model.lasso.less$lambda.1se,
                                                                                  type = "response")
      
      
      #### LASSO + LESSstd scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lasso.lessstd <- cv.glmnet(x = as.matrix(train.n.map.std[, -ncol(train.n.map.std)]),
                                          y = as.numeric(train.n.map.std[, ncol(train.n.map.std)]),
                                          family = "gaussian",
                                          alpha = 1,
                                          foldid = foldid.cv,
                                          type.measure = "mse",
                                          grouped = FALSE)
      
      # Train model
      model.lasso.lessstd <- glmnet(x = as.matrix(train.n.map.std[, -ncol(train.n.map.std)]),
                                 y = as.numeric(train.n.map.std[, ncol(train.n.map.std)]),
                                 intercept = TRUE,
                                 standardize = FALSE,
                                 family = "gaussian",
                                 alpha = 1,
                                 lambda = cv.model.lasso.lessstd$lambda.1se)
      sim.list$numbetas.lasso.lessstd[fold, n] <- sum(coef(model.lasso.lessstd)[-1] != 0)
      
      # Test model
      sim.list$preds.lasso.lessstd[as.numeric(row.names(test.map.std)), n] <- predict.glmnet(object = model.lasso.lessstd,
                                                                                      newx = as.matrix(test.map.std[, -ncol(test.map.std)]),
                                                                                      s = cv.model.lasso.lessstd$lambda.1se,
                                                                                      type = "response")
      
      
      #### LASSO + LESSstd2 scaling ####
      
      # Leave-one-out cross-validation for penalisation parameters lambda
      cv.model.lasso.lessstd2 <- cv.glmnet(x = as.matrix(train.n.map.std2[, -ncol(train.n.map.std2)]),
                                             y = as.numeric(train.n.map.std2[, ncol(train.n.map.std2)]),
                                             family = "gaussian",
                                             alpha = 1,
                                             foldid = foldid.cv,
                                             type.measure = "mse",
                                             grouped = FALSE)
      
      # Train model
      model.lasso.lessstd2 <- glmnet(x = as.matrix(train.n.map.std2[, -ncol(train.n.map.std2)]),
                                    y = as.numeric(train.n.map.std2[, ncol(train.n.map.std2)]),
                                    intercept = TRUE,
                                    standardize = FALSE,
                                    family = "gaussian",
                                    alpha = 1,
                                    lambda = cv.model.lasso.lessstd2$lambda.1se)
      sim.list$numbetas.lasso.lessstd2[fold, n] <- sum(coef(model.lasso.lessstd2)[-1] != 0)
      
      # Test model
      sim.list$preds.lasso.lessstd2[as.numeric(row.names(test.map.std2)), n] <- predict.glmnet(object = model.lasso.lessstd2,
                                                                                             newx = as.matrix(test.map.std2[, -ncol(test.map.std2)]),
                                                                                             s = cv.model.lasso.lessstd2$lambda.1se,
                                                                                             type = "response")
    }
    
    ## Results per fold and per sample size
    # Prediction labels
    sim.list$preds.labs.lrl1.none[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lrl1.none[as.numeric(row.names(test)), ] < 0, lab1, lab2)
    sim.list$preds.labs.lrl1.std[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lrl1.std[as.numeric(row.names(test)), ] < 0, lab1, lab2)
    sim.list$preds.labs.lrl1.less[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lrl1.less[as.numeric(row.names(test)), ] < 0, lab1, lab2)
    sim.list$preds.labs.lrl1.lessstd[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lrl1.lessstd[as.numeric(row.names(test)), ] < 0, lab1, lab2)
    sim.list$preds.labs.lrl1.lessstd2[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lrl1.lessstd2[as.numeric(row.names(test)), ] < 0, lab1, lab2)
    
    sim.list$preds.labs.lasso.none[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lasso.none[as.numeric(row.names(test)), ] < mean(as.numeric(unique(mydata[, ncol(mydata)]))), lab1, lab2)
    sim.list$preds.labs.lasso.std[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lasso.std[as.numeric(row.names(test)), ] < mean(as.numeric(unique(mydata[, ncol(mydata)]))), lab1, lab2)
    sim.list$preds.labs.lasso.less[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lasso.less[as.numeric(row.names(test)), ] < mean(as.numeric(unique(mydata[, ncol(mydata)]))), lab1, lab2)
    sim.list$preds.labs.lasso.lessstd[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lasso.lessstd[as.numeric(row.names(test)), ] < mean(as.numeric(unique(mydata[, ncol(mydata)]))), lab1, lab2)
    sim.list$preds.labs.lasso.lessstd2[as.numeric(row.names(test)), ] <- ifelse(sim.list$preds.lasso.lessstd2[as.numeric(row.names(test)), ] < mean(as.numeric(unique(mydata[, ncol(mydata)]))), lab1, lab2)
    
    
    # AUC
    sim.list$auc.less.none[fold, ] <- apply(sim.list$preds.less.none[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.less.std[fold, ] <- apply(sim.list$preds.less.std[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.less.less[fold, ] <- apply(sim.list$preds.less.less[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.less.lessstd[fold, ] <- apply(sim.list$preds.less.lessstd[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.less.lessstd2[fold, ] <- apply(sim.list$preds.less.lessstd2[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    
    sim.list$auc.svml1.none[fold, ] <- apply(sim.list$preds.svml1.none[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.svml1.std[fold, ] <- apply(sim.list$preds.svml1.std[as.numeric(row.names(test)), ], 
                                           2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                     predictor = x)$auc})
    sim.list$auc.svml1.less[fold, ] <- apply(sim.list$preds.svml1.less[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.svml1.lessstd[fold, ] <- apply(sim.list$preds.svml1.lessstd[as.numeric(row.names(test)), ], 
                                               2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                         predictor = x)$auc})
    sim.list$auc.svml1.lessstd2[fold, ] <- apply(sim.list$preds.svml1.lessstd2[as.numeric(row.names(test)), ], 
                                                2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                          predictor = x)$auc})
    
    sim.list$auc.lrl1.none[fold, ] <- apply(sim.list$preds.lrl1.none[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.lrl1.std[fold, ] <- apply(sim.list$preds.lrl1.std[as.numeric(row.names(test)), ], 
                                           2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                     predictor = x)$auc})
    sim.list$auc.lrl1.less[fold, ] <- apply(sim.list$preds.lrl1.less[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.lrl1.lessstd[fold, ] <- apply(sim.list$preds.lrl1.lessstd[as.numeric(row.names(test)), ], 
                                               2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                         predictor = x)$auc})
    sim.list$auc.lrl1.lessstd2[fold, ] <- apply(sim.list$preds.lrl1.lessstd2[as.numeric(row.names(test)), ], 
                                                2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                          predictor = x)$auc})
    
    sim.list$auc.lasso.none[fold, ] <- apply(sim.list$preds.lasso.none[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.lasso.std[fold, ] <- apply(sim.list$preds.lasso.std[as.numeric(row.names(test)), ], 
                                           2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                     predictor = x)$auc})
    sim.list$auc.lasso.less[fold, ] <- apply(sim.list$preds.lasso.less[as.numeric(row.names(test)), ], 
                                            2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                      predictor = x)$auc})
    sim.list$auc.lasso.lessstd[fold, ] <- apply(sim.list$preds.lasso.lessstd[as.numeric(row.names(test)), ], 
                                               2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                         predictor = x)$auc})
    sim.list$auc.lasso.lessstd2[fold, ] <- apply(sim.list$preds.lasso.lessstd2[as.numeric(row.names(test)), ], 
                                                2, function(x) {pROC::roc(response = test[, ncol(test)],
                                                                          predictor = x)$auc})
    
    # Accuracy
    sim.list$accuracy.less.none[fold, ] <- apply(sim.list$preds.labs.less.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.less.std[fold, ] <- apply(sim.list$preds.labs.less.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.less.less[fold, ] <- apply(sim.list$preds.labs.less.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.less.lessstd[fold, ] <- apply(sim.list$preds.labs.less.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.less.lessstd2[fold, ] <- apply(sim.list$preds.labs.less.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    
    sim.list$accuracy.svml1.none[fold, ] <- apply(sim.list$preds.labs.svml1.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.svml1.std[fold, ] <- apply(sim.list$preds.labs.svml1.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.svml1.less[fold, ] <- apply(sim.list$preds.labs.svml1.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.svml1.lessstd[fold, ] <- apply(sim.list$preds.labs.svml1.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.svml1.lessstd2[fold, ] <- apply(sim.list$preds.labs.svml1.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    
    sim.list$accuracy.lrl1.none[fold, ] <- apply(sim.list$preds.labs.lrl1.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lrl1.std[fold, ] <- apply(sim.list$preds.labs.lrl1.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lrl1.less[fold, ] <- apply(sim.list$preds.labs.lrl1.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lrl1.lessstd[fold, ] <- apply(sim.list$preds.labs.lrl1.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lrl1.lessstd2[fold, ] <- apply(sim.list$preds.labs.lrl1.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    
    sim.list$accuracy.lasso.none[fold, ] <- apply(sim.list$preds.labs.lasso.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lasso.std[fold, ] <- apply(sim.list$preds.labs.lasso.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lasso.less[fold, ] <- apply(sim.list$preds.labs.lasso.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lasso.lessstd[fold, ] <- apply(sim.list$preds.labs.lasso.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    sim.list$accuracy.lasso.lessstd2[fold, ] <- apply(sim.list$preds.labs.lasso.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x == test[, ncol(test)]) * 100})
    
    
    # Generalisation error
    sim.list$generror.less.none[fold, ] <- apply(sim.list$preds.labs.less.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.less.std[fold, ] <- apply(sim.list$preds.labs.less.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.less.less[fold, ] <- apply(sim.list$preds.labs.less.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.less.lessstd[fold, ] <- apply(sim.list$preds.labs.less.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.less.lessstd2[fold, ] <- apply(sim.list$preds.labs.less.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    
    sim.list$generror.svml1.none[fold, ] <- apply(sim.list$preds.labs.svml1.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.svml1.std[fold, ] <- apply(sim.list$preds.labs.svml1.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.svml1.less[fold, ] <- apply(sim.list$preds.labs.svml1.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.svml1.lessstd[fold, ] <- apply(sim.list$preds.labs.svml1.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.svml1.lessstd2[fold, ] <- apply(sim.list$preds.labs.svml1.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    
    sim.list$generror.lrl1.none[fold, ] <- apply(sim.list$preds.labs.lrl1.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lrl1.std[fold, ] <- apply(sim.list$preds.labs.lrl1.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lrl1.less[fold, ] <- apply(sim.list$preds.labs.lrl1.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lrl1.lessstd[fold, ] <- apply(sim.list$preds.labs.lrl1.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lrl1.lessstd2[fold, ] <- apply(sim.list$preds.labs.lrl1.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    
    sim.list$generror.lasso.none[fold, ] <- apply(sim.list$preds.labs.lasso.none[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lasso.std[fold, ] <- apply(sim.list$preds.labs.lasso.std[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lasso.less[fold, ] <- apply(sim.list$preds.labs.lasso.less[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lasso.lessstd[fold, ] <- apply(sim.list$preds.labs.lasso.lessstd[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    sim.list$generror.lasso.lessstd2[fold, ] <- apply(sim.list$preds.labs.lasso.lessstd2[as.numeric(row.names(test)), ], 2, function(x) {mean(x != test[, ncol(test)]) * 100})
    
  }
  sim.list
}  
  
stopImplicitCluster()

save(sim.list, file = "glioma_samplesize_simlist.RData")


## Rbind results from all K repetitions [K * nfold, nsamples]

# list1 <- list(list(a = matrix(1:12, 3, 4),
#                    b = matrix(2, 3, 4),
#                    c = matrix(3, 3, 4)),
#               list(a = matrix(1:12, 3, 4),
#                    b = matrix(2, 3, 4),
#                    c = matrix(3, 3, 4)))
# a.df <- do.call(rbind, lapply(list1, function(k) k[["a"]]))
# mean.a <- colMeans(a.df)
# sd.a <- apply(a.df, 2, sd)

numbetas.less.none <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.less.none"]]))
numbetas.less.std <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.less.std"]]))
numbetas.less.less <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.less.less"]]))
numbetas.less.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.less.lessstd"]]))
numbetas.less.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.less.lessstd2"]]))
numbetas.svml1.none <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.svml1.none"]]))
numbetas.svml1.std <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.svml1.std"]]))
numbetas.svml1.less <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.svml1.less"]]))
numbetas.svml1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.svml1.lessstd"]]))
numbetas.svml1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.svml1.lessstd2"]]))
numbetas.lrl1.none <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lrl1.none"]]))
numbetas.lrl1.std <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lrl1.std"]]))
numbetas.lrl1.less <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lrl1.less"]]))
numbetas.lrl1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lrl1.lessstd"]]))
numbetas.lrl1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lrl1.lessstd2"]]))
numbetas.lasso.none <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lasso.none"]]))
numbetas.lasso.std <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lasso.std"]]))
numbetas.lasso.less <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lasso.less"]]))
numbetas.lasso.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lasso.lessstd"]]))
numbetas.lasso.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["numbetas.lasso.lessstd2"]]))

auc.less.none <- do.call(rbind, lapply(sim.list, function(k) k[["auc.less.none"]]))
auc.less.std <- do.call(rbind, lapply(sim.list, function(k) k[["auc.less.std"]]))
auc.less.less <- do.call(rbind, lapply(sim.list, function(k) k[["auc.less.less"]]))
auc.less.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["auc.less.lessstd"]]))
auc.less.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["auc.less.lessstd2"]]))
auc.svml1.none <- do.call(rbind, lapply(sim.list, function(k) k[["auc.svml1.none"]]))
auc.svml1.std <- do.call(rbind, lapply(sim.list, function(k) k[["auc.svml1.std"]]))
auc.svml1.less <- do.call(rbind, lapply(sim.list, function(k) k[["auc.svml1.less"]]))
auc.svml1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["auc.svml1.lessstd"]]))
auc.svml1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["auc.svml1.lessstd2"]]))
auc.lrl1.none <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lrl1.none"]]))
auc.lrl1.std <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lrl1.std"]]))
auc.lrl1.less <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lrl1.less"]]))
auc.lrl1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lrl1.lessstd"]]))
auc.lrl1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lrl1.lessstd2"]]))
auc.lasso.none <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lasso.none"]]))
auc.lasso.std <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lasso.std"]]))
auc.lasso.less <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lasso.less"]]))
auc.lasso.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lasso.lessstd"]]))
auc.lasso.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["auc.lasso.lessstd2"]]))

accuracy.less.none <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.less.none"]]))
accuracy.less.std <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.less.std"]]))
accuracy.less.less <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.less.less"]]))
accuracy.less.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.less.lessstd"]]))
accuracy.less.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.less.lessstd2"]]))
accuracy.svml1.none <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.svml1.none"]]))
accuracy.svml1.std <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.svml1.std"]]))
accuracy.svml1.less <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.svml1.less"]]))
accuracy.svml1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.svml1.lessstd"]]))
accuracy.svml1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.svml1.lessstd2"]]))
accuracy.lrl1.none <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lrl1.none"]]))
accuracy.lrl1.std <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lrl1.std"]]))
accuracy.lrl1.less <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lrl1.less"]]))
accuracy.lrl1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lrl1.lessstd"]]))
accuracy.lrl1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lrl1.lessstd2"]]))
accuracy.lasso.none <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lasso.none"]]))
accuracy.lasso.std <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lasso.std"]]))
accuracy.lasso.less <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lasso.less"]]))
accuracy.lasso.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lasso.lessstd"]]))
accuracy.lasso.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["accuracy.lasso.lessstd2"]]))

generror.less.none <- do.call(rbind, lapply(sim.list, function(k) k[["generror.less.none"]]))
generror.less.std <- do.call(rbind, lapply(sim.list, function(k) k[["generror.less.std"]]))
generror.less.less <- do.call(rbind, lapply(sim.list, function(k) k[["generror.less.less"]]))
generror.less.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["generror.less.lessstd"]]))
generror.less.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["generror.less.lessstd2"]]))
generror.svml1.none <- do.call(rbind, lapply(sim.list, function(k) k[["generror.svml1.none"]]))
generror.svml1.std <- do.call(rbind, lapply(sim.list, function(k) k[["generror.svml1.std"]]))
generror.svml1.less <- do.call(rbind, lapply(sim.list, function(k) k[["generror.svml1.less"]]))
generror.svml1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["generror.svml1.lessstd"]]))
generror.svml1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["generror.svml1.lessstd2"]]))
generror.lrl1.none <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lrl1.none"]]))
generror.lrl1.std <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lrl1.std"]]))
generror.lrl1.less <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lrl1.less"]]))
generror.lrl1.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lrl1.lessstd"]]))
generror.lrl1.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lrl1.lessstd2"]]))
generror.lasso.none <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lasso.none"]]))
generror.lasso.std <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lasso.std"]]))
generror.lasso.less <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lasso.less"]]))
generror.lasso.lessstd <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lasso.lessstd"]]))
generror.lasso.lessstd2 <- do.call(rbind, lapply(sim.list, function(k) k[["generror.lasso.lessstd2"]]))


# Mean over all K repetitions [1, nsamples]
mean.numbetas.less.none <- colMeans(numbetas.less.none)
mean.numbetas.less.std <- colMeans(numbetas.less.std)
mean.numbetas.less.less <- colMeans(numbetas.less.less)
mean.numbetas.less.lessstd <- colMeans(numbetas.less.lessstd)
mean.numbetas.less.lessstd2 <- colMeans(numbetas.less.lessstd2)
mean.numbetas.svml1.none <- colMeans(numbetas.svml1.none)
mean.numbetas.svml1.std <- colMeans(numbetas.svml1.std)
mean.numbetas.svml1.less <- colMeans(numbetas.svml1.less)
mean.numbetas.svml1.lessstd <- colMeans(numbetas.svml1.lessstd)
mean.numbetas.svml1.lessstd2 <- colMeans(numbetas.svml1.lessstd2)
mean.numbetas.lrl1.none <- colMeans(numbetas.lrl1.none)
mean.numbetas.lrl1.std <- colMeans(numbetas.lrl1.std)
mean.numbetas.lrl1.less <- colMeans(numbetas.lrl1.less)
mean.numbetas.lrl1.lessstd <- colMeans(numbetas.lrl1.lessstd)
mean.numbetas.lrl1.lessstd2 <- colMeans(numbetas.lrl1.lessstd2)
mean.numbetas.lasso.none <- colMeans(numbetas.lasso.none)
mean.numbetas.lasso.std <- colMeans(numbetas.lasso.std)
mean.numbetas.lasso.less <- colMeans(numbetas.lasso.less)
mean.numbetas.lasso.lessstd <- colMeans(numbetas.lasso.lessstd)
mean.numbetas.lasso.lessstd2 <- colMeans(numbetas.lasso.lessstd2)

mean.auc.less.none <- colMeans(auc.less.none)
mean.auc.less.std <- colMeans(auc.less.std)
mean.auc.less.less <- colMeans(auc.less.less)
mean.auc.less.lessstd <- colMeans(auc.less.lessstd)
mean.auc.less.lessstd2 <- colMeans(auc.less.lessstd2)
mean.auc.svml1.none <- colMeans(auc.svml1.none)
mean.auc.svml1.std <- colMeans(auc.svml1.std)
mean.auc.svml1.less <- colMeans(auc.svml1.less)
mean.auc.svml1.lessstd <- colMeans(auc.svml1.lessstd)
mean.auc.svml1.lessstd2 <- colMeans(auc.svml1.lessstd2)
mean.auc.lrl1.none <- colMeans(auc.lrl1.none)
mean.auc.lrl1.std <- colMeans(auc.lrl1.std)
mean.auc.lrl1.less <- colMeans(auc.lrl1.less)
mean.auc.lrl1.lessstd <- colMeans(auc.lrl1.lessstd)
mean.auc.lrl1.lessstd2 <- colMeans(auc.lrl1.lessstd2)
mean.auc.lasso.none <- colMeans(auc.lasso.none)
mean.auc.lasso.std <- colMeans(auc.lasso.std)
mean.auc.lasso.less <- colMeans(auc.lasso.less)
mean.auc.lasso.lessstd <- colMeans(auc.lasso.lessstd)
mean.auc.lasso.lessstd2 <- colMeans(auc.lasso.lessstd2)

mean.accuracy.less.none <- colMeans(accuracy.less.none)
mean.accuracy.less.std <- colMeans(accuracy.less.std)
mean.accuracy.less.less <- colMeans(accuracy.less.less)
mean.accuracy.less.lessstd <- colMeans(accuracy.less.lessstd)
mean.accuracy.less.lessstd2 <- colMeans(accuracy.less.lessstd2)
mean.accuracy.svml1.none <- colMeans(accuracy.svml1.none)
mean.accuracy.svml1.std <- colMeans(accuracy.svml1.std)
mean.accuracy.svml1.less <- colMeans(accuracy.svml1.less)
mean.accuracy.svml1.lessstd <- colMeans(accuracy.svml1.lessstd)
mean.accuracy.svml1.lessstd2 <- colMeans(accuracy.svml1.lessstd2)
mean.accuracy.lrl1.none <- colMeans(accuracy.lrl1.none)
mean.accuracy.lrl1.std <- colMeans(accuracy.lrl1.std)
mean.accuracy.lrl1.less <- colMeans(accuracy.lrl1.less)
mean.accuracy.lrl1.lessstd <- colMeans(accuracy.lrl1.lessstd)
mean.accuracy.lrl1.lessstd2 <- colMeans(accuracy.lrl1.lessstd2)
mean.accuracy.lasso.none <- colMeans(accuracy.lasso.none)
mean.accuracy.lasso.std <- colMeans(accuracy.lasso.std)
mean.accuracy.lasso.less <- colMeans(accuracy.lasso.less)
mean.accuracy.lasso.lessstd <- colMeans(accuracy.lasso.lessstd)
mean.accuracy.lasso.lessstd2 <- colMeans(accuracy.lasso.lessstd2)

mean.generror.less.none <- colMeans(generror.less.none)
mean.generror.less.std <- colMeans(generror.less.std)
mean.generror.less.less <- colMeans(generror.less.less)
mean.generror.less.lessstd <- colMeans(generror.less.lessstd)
mean.generror.less.lessstd2 <- colMeans(generror.less.lessstd2)
mean.generror.svml1.none <- colMeans(generror.svml1.none)
mean.generror.svml1.std <- colMeans(generror.svml1.std)
mean.generror.svml1.less <- colMeans(generror.svml1.less)
mean.generror.svml1.lessstd <- colMeans(generror.svml1.lessstd)
mean.generror.svml1.lessstd2 <- colMeans(generror.svml1.lessstd2)
mean.generror.lrl1.none <- colMeans(generror.lrl1.none)
mean.generror.lrl1.std <- colMeans(generror.lrl1.std)
mean.generror.lrl1.less <- colMeans(generror.lrl1.less)
mean.generror.lrl1.lessstd <- colMeans(generror.lrl1.lessstd)
mean.generror.lrl1.lessstd2 <- colMeans(generror.lrl1.lessstd2)
mean.generror.lasso.none <- colMeans(generror.lasso.none)
mean.generror.lasso.std <- colMeans(generror.lasso.std)
mean.generror.lasso.less <- colMeans(generror.lasso.less)
mean.generror.lasso.lessstd <- colMeans(generror.lasso.lessstd)
mean.generror.lasso.lessstd2 <- colMeans(generror.lasso.lessstd2)


# Sd over all K repetitions [1, nsamples]
sd.numbetas.less.none <- apply(numbetas.less.none, 2, sd)
sd.numbetas.less.std <- apply(numbetas.less.std, 2, sd)
sd.numbetas.less.less <- apply(numbetas.less.less, 2, sd)
sd.numbetas.less.lessstd <- apply(numbetas.less.lessstd, 2, sd)
sd.numbetas.less.lessstd2 <- apply(numbetas.less.lessstd2, 2, sd)
sd.numbetas.svml1.none <- apply(numbetas.svml1.none, 2, sd)
sd.numbetas.svml1.std <- apply(numbetas.svml1.std, 2, sd)
sd.numbetas.svml1.less <- apply(numbetas.svml1.less, 2, sd)
sd.numbetas.svml1.lessstd <- apply(numbetas.svml1.lessstd, 2, sd)
sd.numbetas.svml1.lessstd2 <- apply(numbetas.svml1.lessstd2, 2, sd)
sd.numbetas.lrl1.none <- apply(numbetas.lrl1.none, 2, sd)
sd.numbetas.lrl1.std <- apply(numbetas.lrl1.std, 2, sd)
sd.numbetas.lrl1.less <- apply(numbetas.lrl1.less, 2, sd)
sd.numbetas.lrl1.lessstd <- apply(numbetas.lrl1.lessstd, 2, sd)
sd.numbetas.lrl1.lessstd2 <- apply(numbetas.lrl1.lessstd2, 2, sd)
sd.numbetas.lasso.none <- apply(numbetas.lasso.none, 2, sd)
sd.numbetas.lasso.std <- apply(numbetas.lasso.std, 2, sd)
sd.numbetas.lasso.less <- apply(numbetas.lasso.less, 2, sd)
sd.numbetas.lasso.lessstd <- apply(numbetas.lasso.lessstd, 2, sd)
sd.numbetas.lasso.lessstd2 <- apply(numbetas.lasso.lessstd2, 2, sd)

sd.auc.less.none <- apply(auc.less.none, 2, sd)
sd.auc.less.std <- apply(auc.less.std, 2, sd)
sd.auc.less.less <- apply(auc.less.less, 2, sd)
sd.auc.less.lessstd <- apply(auc.less.lessstd, 2, sd)
sd.auc.less.lessstd2 <- apply(auc.less.lessstd2, 2, sd)
sd.auc.svml1.none <- apply(auc.svml1.none, 2, sd)
sd.auc.svml1.std <- apply(auc.svml1.std, 2, sd)
sd.auc.svml1.less <- apply(auc.svml1.less, 2, sd)
sd.auc.svml1.lessstd <- apply(auc.svml1.lessstd, 2, sd)
sd.auc.svml1.lessstd2 <- apply(auc.svml1.lessstd2, 2, sd)
sd.auc.lrl1.none <- apply(auc.lrl1.none, 2, sd)
sd.auc.lrl1.std <- apply(auc.lrl1.std, 2, sd)
sd.auc.lrl1.less <- apply(auc.lrl1.less, 2, sd)
sd.auc.lrl1.lessstd <- apply(auc.lrl1.lessstd, 2, sd)
sd.auc.lrl1.lessstd2 <- apply(auc.lrl1.lessstd2, 2, sd)
sd.auc.lasso.none <- apply(auc.lasso.none, 2, sd)
sd.auc.lasso.std <- apply(auc.lasso.std, 2, sd)
sd.auc.lasso.less <- apply(auc.lasso.less, 2, sd)
sd.auc.lasso.lessstd <- apply(auc.lasso.lessstd, 2, sd)
sd.auc.lasso.lessstd2 <- apply(auc.lasso.lessstd2, 2, sd)

sd.accuracy.less.none <- apply(accuracy.less.none, 2, sd)
sd.accuracy.less.std <- apply(accuracy.less.std, 2, sd)
sd.accuracy.less.less <- apply(accuracy.less.less, 2, sd)
sd.accuracy.less.lessstd <- apply(accuracy.less.lessstd, 2, sd)
sd.accuracy.less.lessstd2 <- apply(accuracy.less.lessstd2, 2, sd)
sd.accuracy.svml1.none <- apply(accuracy.svml1.none, 2, sd)
sd.accuracy.svml1.std <- apply(accuracy.svml1.std, 2, sd)
sd.accuracy.svml1.less <- apply(accuracy.svml1.less, 2, sd)
sd.accuracy.svml1.lessstd <- apply(accuracy.svml1.lessstd, 2, sd)
sd.accuracy.svml1.lessstd2 <- apply(accuracy.svml1.lessstd2, 2, sd)
sd.accuracy.lrl1.none <- apply(accuracy.lrl1.none, 2, sd)
sd.accuracy.lrl1.std <- apply(accuracy.lrl1.std, 2, sd)
sd.accuracy.lrl1.less <- apply(accuracy.lrl1.less, 2, sd)
sd.accuracy.lrl1.lessstd <- apply(accuracy.lrl1.lessstd, 2, sd)
sd.accuracy.lrl1.lessstd2 <- apply(accuracy.lrl1.lessstd2, 2, sd)
sd.accuracy.lasso.none <- apply(accuracy.lasso.none, 2, sd)
sd.accuracy.lasso.std <- apply(accuracy.lasso.std, 2, sd)
sd.accuracy.lasso.less <- apply(accuracy.lasso.less, 2, sd)
sd.accuracy.lasso.lessstd <- apply(accuracy.lasso.lessstd, 2, sd)
sd.accuracy.lasso.lessstd2 <- apply(accuracy.lasso.lessstd2, 2, sd)

sd.generror.less.none <- apply(generror.less.none, 2, sd)
sd.generror.less.std <- apply(generror.less.std, 2, sd)
sd.generror.less.less <- apply(generror.less.less, 2, sd)
sd.generror.less.lessstd <- apply(generror.less.lessstd, 2, sd)
sd.generror.less.lessstd2 <- apply(generror.less.lessstd2, 2, sd)
sd.generror.svml1.none <- apply(generror.svml1.none, 2, sd)
sd.generror.svml1.std <- apply(generror.svml1.std, 2, sd)
sd.generror.svml1.less <- apply(generror.svml1.less, 2, sd)
sd.generror.svml1.lessstd <- apply(generror.svml1.lessstd, 2, sd)
sd.generror.svml1.lessstd2 <- apply(generror.svml1.lessstd2, 2, sd)
sd.generror.lrl1.none <- apply(generror.lrl1.none, 2, sd)
sd.generror.lrl1.std <- apply(generror.lrl1.std, 2, sd)
sd.generror.lrl1.less <- apply(generror.lrl1.less, 2, sd)
sd.generror.lrl1.lessstd <- apply(generror.lrl1.lessstd, 2, sd)
sd.generror.lrl1.lessstd2 <- apply(generror.lrl1.lessstd2, 2, sd)
sd.generror.lasso.none <- apply(generror.lasso.none, 2, sd)
sd.generror.lasso.std <- apply(generror.lasso.std, 2, sd)
sd.generror.lasso.less <- apply(generror.lasso.less, 2, sd)
sd.generror.lasso.lessstd <- apply(generror.lasso.lessstd, 2, sd)
sd.generror.lasso.lessstd2 <- apply(generror.lasso.lessstd2, 2, sd)


#### Summary ####
plots.mydata <- data.frame(SampleSize = rep(N, 20),
                           AUC.mean = c(mean.auc.less.none,
                                        mean.auc.less.std,
                                        mean.auc.less.less,
                                        mean.auc.less.lessstd,
                                        mean.auc.less.lessstd2,
                                        mean.auc.svml1.none,
                                        mean.auc.svml1.std,
                                        mean.auc.svml1.less,
                                        mean.auc.svml1.lessstd,
                                        mean.auc.svml1.lessstd2,
                                        mean.auc.lrl1.none,
                                        mean.auc.lrl1.std,
                                        mean.auc.lrl1.less,
                                        mean.auc.lrl1.lessstd,
                                        mean.auc.lrl1.lessstd2,
                                        mean.auc.lasso.none,
                                        mean.auc.lasso.std,
                                        mean.auc.lasso.less,
                                        mean.auc.lasso.lessstd,
                                        mean.auc.lasso.lessstd2),
                           AUC.sd = c(sd.auc.less.none,
                                      sd.auc.less.std,
                                      sd.auc.less.less,
                                      sd.auc.less.lessstd,
                                      sd.auc.less.lessstd2,
                                      sd.auc.svml1.none,
                                      sd.auc.svml1.std,
                                      sd.auc.svml1.less,
                                      sd.auc.svml1.lessstd,
                                      sd.auc.svml1.lessstd2,
                                      sd.auc.lrl1.none,
                                      sd.auc.lrl1.std,
                                      sd.auc.lrl1.less,
                                      sd.auc.lrl1.lessstd,
                                      sd.auc.lrl1.lessstd2,
                                      sd.auc.lasso.none,
                                      sd.auc.lasso.std,
                                      sd.auc.lasso.less,
                                      sd.auc.lasso.lessstd,
                                      sd.auc.lasso.lessstd2),
                           Accuracy.mean = c(mean.accuracy.less.none,
                                             mean.accuracy.less.std,
                                             mean.accuracy.less.less,
                                             mean.accuracy.less.lessstd,
                                             mean.accuracy.less.lessstd2,
                                             mean.accuracy.svml1.none,
                                             mean.accuracy.svml1.std,
                                             mean.accuracy.svml1.less,
                                             mean.accuracy.svml1.lessstd,
                                             mean.accuracy.svml1.lessstd2,
                                             mean.accuracy.lrl1.none,
                                             mean.accuracy.lrl1.std,
                                             mean.accuracy.lrl1.less,
                                             mean.accuracy.lrl1.lessstd,
                                             mean.accuracy.lrl1.lessstd2,
                                             mean.accuracy.lasso.none,
                                             mean.accuracy.lasso.std,
                                             mean.accuracy.lasso.less,
                                             mean.accuracy.lasso.lessstd,
                                             mean.accuracy.lasso.lessstd2),
                           Accuracy.sd = c(sd.accuracy.less.none,
                                           sd.accuracy.less.std,
                                           sd.accuracy.less.less,
                                           sd.accuracy.less.lessstd,
                                           sd.accuracy.less.lessstd2,
                                           sd.accuracy.svml1.none,
                                           sd.accuracy.svml1.std,
                                           sd.accuracy.svml1.less,
                                           sd.accuracy.svml1.lessstd,
                                           sd.accuracy.svml1.lessstd2,
                                           sd.accuracy.lrl1.none,
                                           sd.accuracy.lrl1.std,
                                           sd.accuracy.lrl1.less,
                                           sd.accuracy.lrl1.lessstd,
                                           sd.accuracy.lrl1.lessstd2,
                                           sd.accuracy.lasso.none,
                                           sd.accuracy.lasso.std,
                                           sd.accuracy.lasso.less,
                                           sd.accuracy.lasso.lessstd,
                                           sd.accuracy.lasso.lessstd2),
                           Generalisation.Error.mean = c(mean.generror.less.none,
                                                         mean.generror.less.std,
                                                         mean.generror.less.less,
                                                         mean.generror.less.lessstd,
                                                         mean.generror.less.lessstd2,
                                                         mean.generror.svml1.none,
                                                         mean.generror.svml1.std,
                                                         mean.generror.svml1.less,
                                                         mean.generror.svml1.lessstd,
                                                         mean.generror.svml1.lessstd2,
                                                         mean.generror.lrl1.none,
                                                         mean.generror.lrl1.std,
                                                         mean.generror.lrl1.less,
                                                         mean.generror.lrl1.lessstd,
                                                         mean.generror.lrl1.lessstd2,
                                                         mean.generror.lasso.none,
                                                         mean.generror.lasso.std,
                                                         mean.generror.lasso.less,
                                                         mean.generror.lasso.lessstd,
                                                         mean.generror.lasso.lessstd2),
                           Generalisation.Error.sd = c(sd.generror.less.none,
                                                       sd.generror.less.std,
                                                       sd.generror.less.less,
                                                       sd.generror.less.lessstd,
                                                       sd.generror.less.lessstd2,
                                                       sd.generror.svml1.none,
                                                       sd.generror.svml1.std,
                                                       sd.generror.svml1.less,
                                                       sd.generror.svml1.lessstd,
                                                       sd.generror.svml1.lessstd2,
                                                       sd.generror.lrl1.none,
                                                       sd.generror.lrl1.std,
                                                       sd.generror.lrl1.less,
                                                       sd.generror.lrl1.lessstd,
                                                       sd.generror.lrl1.lessstd2,
                                                       sd.generror.lasso.none,
                                                       sd.generror.lasso.std,
                                                       sd.generror.lasso.less,
                                                       sd.generror.lasso.lessstd,
                                                       sd.generror.lasso.lessstd2),
                           Dimensionality.mean = c(mean.numbetas.less.none,
                                                   mean.numbetas.less.std,
                                                   mean.numbetas.less.less,
                                                   mean.numbetas.less.lessstd,
                                                   mean.numbetas.less.lessstd2,
                                                   mean.numbetas.svml1.none,
                                                   mean.numbetas.svml1.std,
                                                   mean.numbetas.svml1.less,
                                                   mean.numbetas.svml1.lessstd,
                                                   mean.numbetas.svml1.lessstd2,
                                                   mean.numbetas.lrl1.none,
                                                   mean.numbetas.lrl1.std,
                                                   mean.numbetas.lrl1.less,
                                                   mean.numbetas.lrl1.lessstd,
                                                   mean.numbetas.lrl1.lessstd2,
                                                   mean.numbetas.lasso.none,
                                                   mean.numbetas.lasso.std,
                                                   mean.numbetas.lasso.less,
                                                   mean.numbetas.lasso.lessstd,
                                                   mean.numbetas.lasso.lessstd2),
                           Dimensionality.sd = c(sd.numbetas.less.none,
                                                 sd.numbetas.less.std,
                                                 sd.numbetas.less.less,
                                                 sd.numbetas.less.lessstd,
                                                 sd.numbetas.less.lessstd2,
                                                 sd.numbetas.svml1.none,
                                                 sd.numbetas.svml1.std,
                                                 sd.numbetas.svml1.less,
                                                 sd.numbetas.svml1.lessstd,
                                                 sd.numbetas.svml1.lessstd2,
                                                 sd.numbetas.lrl1.none,
                                                 sd.numbetas.lrl1.std,
                                                 sd.numbetas.lrl1.less,
                                                 sd.numbetas.lrl1.lessstd,
                                                 sd.numbetas.lrl1.lessstd2,
                                                 sd.numbetas.lasso.none,
                                                 sd.numbetas.lasso.std,
                                                 sd.numbetas.lasso.less,
                                                 sd.numbetas.lasso.lessstd,
                                                 sd.numbetas.lasso.lessstd2),
                           Method = c(rep("LESS", length(N) * 5),
                                      rep("SVM", length(N) * 5),
                                      rep("LogReg", length(N) * 5),
                                      rep("LASSO", length(N) * 5)),
                           Scaling = rep(c(rep("none", length(N)),
                                           rep("std", length(N)),
                                           rep("less", length(N)),
                                           rep("lessstd", length(N)),
                                           rep("lessstd2", length(N))), 4))

write.csv(plots.mydata, "glioma_samplesize_results.csv", row.names = FALSE)


#### Plot results ####

# # Colors
groups <- 4
cols <- hcl(h = seq(15, 375, length = groups + 1), l = 65, c = 100)[1:groups]
# plot(1:groups, pch = 16, cex = 7, col = cols)
# plot(c(2:6, 9:13, 17:21, 24:28), pch = 16, cex = 7, col = cols[c(2:6, 9:13, 17:21, 24:28)])
# cols

# Relevel factors for correct order in plots
plots.mydata$Method <- factor(plots.mydata$Method, levels = c("LASSO", "LogReg", "SVM", "LESS"))
plots.mydata$Scaling <- factor(plots.mydata$Scaling, levels = c("none", "std", "less", "lessstd", "lessstd2"))

# Remove LESS_none and LESS_std
plots.mydata <- subset(plots.mydata, 
                       !((plots.mydata$Method == "LESS" & plots.mydata$Scaling == "none") |
                           (plots.mydata$Method == "LESS" & plots.mydata$Scaling == "std")))


ggsave("glioma_samplesize_numbetas.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = Dimensionality.mean, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Number of selected variables for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("Model dimensionality") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave("other plots/glioma_samplesize_numbetas_sd.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = Dimensionality.sd, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Standard deviation of number of selected variables for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("Standard deviation of model dimensionality") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")


ggsave("glioma_samplesize_AUC.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = AUC.mean, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Test AUC for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave("other plots/glioma_samplesize_AUC_sd.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = AUC.sd, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Standard deviation of test AUC for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("Standard deviation of AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")


ggsave("glioma_samplesize_accuracy.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = Accuracy.mean, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Test accuracy for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("Accuracy (%)") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave("other plots/glioma_samplesize_accuracy_sd.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = Accuracy.sd, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Standard deviation of test accuracy for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("Standard deviation of accuracy") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")


ggsave("glioma_samplesize_generror.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = Generalisation.Error.mean, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Generalisation error for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("Generalisation error (%)") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave("other plots/glioma_samplesize_generror_sd.png",
       ggplot(plots.mydata, aes(x = SampleSize)) +
         geom_line(aes(y = Generalisation.Error.sd, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle("Standard deviation of generalisation error for increasing sample size (glioma data)") +
         xlab("Sample size") +
         ylab("Standard deviation of generalisation error") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

send_telegram_message(text = "The glioma script with increasing sample size is finished!",
                      chat_id = "441084295",
                      bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")

#### END ####