rm(list = ls())

library(foreach)
library(parallel)
library(iterators)
library(doParallel)
library(glmnet)
library(lpSolve)
library(ggplot2)
library(pROC)
library(R.matlab)
library(LiblineaR)
library(telegram)
library(latex2exp)
source("Scripts/lesstwoc_mv.R")
source("Scripts/Telegram.R")

mydata <- read.csv("Data/leukemia_small2.txt", sep = "", stringsAsFactors = FALSE)
dim(mydata) # 72, 3572
table(mydata[, ncol(mydata)]) # 47, 25

ncores <- 50   # number of parallel cpu cores
K <- 100       # number of repetitions to average over
nfolds <- 10   # number of folds for results

# Create list for results [nsamples, nfolds, K]
sim.list <- list(C.less.none = numeric(nfolds),
                 C.less.std = numeric(nfolds),
                 C.less.less = numeric(nfolds),
                 C.less.lessstd = numeric(nfolds),
                 C.less.lessstd2 = numeric(nfolds),
                 
                 C.svml1.none = numeric(nfolds),
                 C.svml1.std = numeric(nfolds),
                 C.svml1.less = numeric(nfolds),
                 C.svml1.lessstd = numeric(nfolds),
                 C.svml1.lessstd2 = numeric(nfolds),
                 
                 numbetas.less.none = numeric(nfolds),
                 numbetas.less.std = numeric(nfolds),
                 numbetas.less.less = numeric(nfolds),
                 numbetas.less.lessstd = numeric(nfolds),
                 numbetas.less.lessstd2 = numeric(nfolds),
                 numbetas.svml1.none = numeric(nfolds),
                 numbetas.svml1.std = numeric(nfolds),
                 numbetas.svml1.less = numeric(nfolds),
                 numbetas.svml1.lessstd = numeric(nfolds),
                 numbetas.svml1.lessstd2 = numeric(nfolds),
                 numbetas.lrl1.none = numeric(nfolds),
                 numbetas.lrl1.std = numeric(nfolds),
                 numbetas.lrl1.less = numeric(nfolds),
                 numbetas.lrl1.lessstd = numeric(nfolds),
                 numbetas.lrl1.lessstd2 = numeric(nfolds),
                 numbetas.lasso.none = numeric(nfolds),
                 numbetas.lasso.std = numeric(nfolds),
                 numbetas.lasso.less = numeric(nfolds),
                 numbetas.lasso.lessstd = numeric(nfolds),
                 numbetas.lasso.lessstd2 = numeric(nfolds),
                 
                 auc.less.none = numeric(nfolds),
                 auc.less.std = numeric(nfolds),
                 auc.less.less = numeric(nfolds),
                 auc.less.lessstd = numeric(nfolds),
                 auc.less.lessstd2 = numeric(nfolds),
                 auc.svml1.none = numeric(nfolds),
                 auc.svml1.std = numeric(nfolds),
                 auc.svml1.less = numeric(nfolds),
                 auc.svml1.lessstd = numeric(nfolds),
                 auc.svml1.lessstd2 = numeric(nfolds),
                 auc.lrl1.none = numeric(nfolds),
                 auc.lrl1.std = numeric(nfolds),
                 auc.lrl1.less = numeric(nfolds),
                 auc.lrl1.lessstd = numeric(nfolds),
                 auc.lrl1.lessstd2 = numeric(nfolds),
                 auc.lasso.none = numeric(nfolds),
                 auc.lasso.std = numeric(nfolds),
                 auc.lasso.less = numeric(nfolds),
                 auc.lasso.lessstd = numeric(nfolds),
                 auc.lasso.lessstd2 = numeric(nfolds),
                 
                 accuracy.less.none = numeric(nfolds),
                 accuracy.less.std = numeric(nfolds),
                 accuracy.less.less = numeric(nfolds),
                 accuracy.less.lessstd = numeric(nfolds),
                 accuracy.less.lessstd2 = numeric(nfolds),
                 accuracy.svml1.none = numeric(nfolds),
                 accuracy.svml1.std = numeric(nfolds),
                 accuracy.svml1.less = numeric(nfolds),
                 accuracy.svml1.lessstd = numeric(nfolds),
                 accuracy.svml1.lessstd2 = numeric(nfolds),
                 accuracy.lrl1.none = numeric(nfolds),
                 accuracy.lrl1.std = numeric(nfolds),
                 accuracy.lrl1.less = numeric(nfolds),
                 accuracy.lrl1.lessstd = numeric(nfolds),
                 accuracy.lrl1.lessstd2 = numeric(nfolds),
                 accuracy.lasso.none = numeric(nfolds),
                 accuracy.lasso.std = numeric(nfolds),
                 accuracy.lasso.less = numeric(nfolds),
                 accuracy.lasso.lessstd = numeric(nfolds),
                 accuracy.lasso.lessstd2 = numeric(nfolds),
                 
                 generror.less.none = numeric(nfolds),
                 generror.less.std = numeric(nfolds),
                 generror.less.less = numeric(nfolds),
                 generror.less.lessstd = numeric(nfolds),
                 generror.less.lessstd2 = numeric(nfolds),
                 generror.svml1.none = numeric(nfolds),
                 generror.svml1.std = numeric(nfolds),
                 generror.svml1.less = numeric(nfolds),
                 generror.svml1.lessstd = numeric(nfolds),
                 generror.svml1.lessstd2 = numeric(nfolds),
                 generror.lrl1.none = numeric(nfolds),
                 generror.lrl1.std = numeric(nfolds),
                 generror.lrl1.less = numeric(nfolds),
                 generror.lrl1.lessstd = numeric(nfolds),
                 generror.lrl1.lessstd2 = numeric(nfolds),
                 generror.lasso.none = numeric(nfolds),
                 generror.lasso.std = numeric(nfolds),
                 generror.lasso.less = numeric(nfolds),
                 generror.lasso.lessstd = numeric(nfolds),
                 generror.lasso.lessstd2 = numeric(nfolds))

# Go parallel
registerDoParallel(cores = ncores)

sim.list <- foreach(k = 1:K) %dopar% {
  print(paste("Repetition", k, "/", K))
  
  set.seed(0952702 + k)
  ncores <- 50   # number of parallel cpu cores
  K <- 100       # number of repetitions to average over
  nfolds <- 10   # number of folds for results

  # Load packages
  library(foreach)
  library(parallel)
  library(iterators)
  library(doParallel)
  library(glmnet)
  library(lpSolve)
  library(ggplot2)
  library(pROC)
  library(R.matlab)
  library(LiblineaR)
  library(telegram)
  library(latex2exp)
  source("Scripts/lesstwoc_mv.R")
  source("Scripts/Telegram.R")
  
  # Load data
  mydata <- read.csv("Data/leukemia_small2.txt", sep = "", stringsAsFactors = FALSE)
  dim(mydata) # 72, 3572
  
  mydata[, ncol(mydata)] <- as.factor(mydata[, ncol(mydata)])
  lab1 <- levels(mydata[, ncol(mydata)])[1] # "ALL"
  lab2 <- levels(mydata[, ncol(mydata)])[2] # "AML"
  
  # Cross-validation settings
  # Specify folds for class 1
  c1 <- as.numeric(row.names(mydata[mydata[, ncol(mydata)] == lab1, ]))
  fold.id.c1 <- sample(rep(seq(nfolds), length = length(c1)))
  # Specify folds for class 2
  c2 <- as.numeric(row.names(mydata[mydata[, ncol(mydata)] == lab2, ]))
  fold.id.c2 <- sample(rep(seq(nfolds), length = length(c2)))
  
  # 10 fold cross-validation (outer loop)
  for (fold in 1:nfolds) {
    print(paste("Fold", fold, "/", nfolds))
    train <- rbind(mydata[c1[fold.id.c1 != fold], ],
                   mydata[c2[fold.id.c2 != fold], ])
    
    test <- rbind(mydata[c1[fold.id.c1 == fold], ],
                  mydata[c2[fold.id.c2 == fold], ])
    
    ## Data scaling based on train set
    
    # Map data for standardisation
    train.std <- train
    for (j in 2:ncol(train) - 1) {
      train.std[, j] <- (train[, j] - mean(train[, j])) / sd(train[, j]) ^ as.logical(sd(train[, j]))
    }
    
    test.std <- test
    for (j in 2:ncol(train) - 1) {
      test.std[, j] <- (test[, j] - mean(train[, j])) / sd(train[, j]) ^ as.logical(sd(train[, j]))
    }
    
    # Map data for LESS scaling
    M.train <- matrix(0.0, nrow = length(levels(train[, ncol(train)])), ncol = ncol(train) - 1)
    M.train[1, ] <- colMeans(train[train[, ncol(train)] == lab1, -ncol(train)])
    M.train[2, ] <- colMeans(train[train[, ncol(train)] == lab2, -ncol(train)])
    
    train.map <- train
    train.map[, -ncol(train.map)] <- mapmeans(DF = train[, -ncol(train)], M = M.train)
    
    test.map <- test
    test.map[, -ncol(test.map)] <- mapmeans(DF = test[, -ncol(test)], M = M.train)
    
    # Map data for LESSstd scaling
    S.train <- matrix(0.0, nrow = length(levels(train[, ncol(train)])), ncol = ncol(train) - 1)
    S.train[1, ] <- apply(train[train[, ncol(train)] == lab1, -ncol(train)], 2, var)
    S.train[2, ] <- apply(train[train[, ncol(train)] == lab2, -ncol(train)], 2, var)
    S.train <- ifelse(S.train == 0, unique(sort(S.train))[2], S.train) # When var = 0
    
    train.map.std <- train
    train.map.std[, -ncol(train.map.std)] <- mapmeansstd(DF = train[, -ncol(train)], 
                                                             M = M.train, 
                                                             S = S.train)
    
    test.map.std <- test
    test.map.std[, -ncol(test.map.std)] <- mapmeansstd(DF = test[, -ncol(test)], 
                                                       M = M.train, 
                                                       S = S.train)
    
    # Map data for LESSstd2 scaling
    S.train2 <- matrix(apply(rbind(train[train[, ncol(train)] == lab1, -ncol(train)] -
                                       matrix(rep(M.train[1, ],
                                                  times = sum(train[, ncol(train)] == lab1)),
                                              nrow = sum(train[, ncol(train)] == lab1),
                                              byrow = TRUE),
                                     train[train[, ncol(train)] == lab2, -ncol(train)] -
                                       matrix(rep(M.train[2, ],
                                                  times = sum(train[, ncol(train)] == lab2)),
                                              nrow = sum(train[, ncol(train)] == lab2),
                                              byrow = TRUE)),
                               2, var),
                         nrow = 1, ncol = ncol(train) - 1)
    S.train2 <- ifelse(S.train2 == 0, unique(sort(S.train2))[2], S.train2) # When var = 0
    
    train.map.std2 <- train
    train.map.std2[, -ncol(train)] <- mapmeansstd2(DF = train[, -ncol(train)], 
                                                       M = M.train, 
                                                       S = S.train2)
    
    test.map.std2 <- test
    test.map.std2[, -ncol(test)] <- mapmeansstd2(DF = test[, -ncol(test)], M = M.train, S = S.train2)
    
    # fold id per class (inner loop for cross validation)
    c1.2 <- as.numeric(row.names(train[train[, ncol(train)] == lab1, ]))
    foldid.c1.2 <- sample(rep(seq(nfolds), length = length(c1.2)))
    c2.2 <- as.numeric(row.names(train[train[, ncol(train)] == lab2, ]))
    foldid.c2.2 <- sample(rep(seq(nfolds), length = length(c2.2)))
    foldid.cv <- c(foldid.c1.2, foldid.c2.2)
    
    
    
    #### LESS ##################################################################

    ### LESS + no scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.less.none <- 10^seq(-4, 1, length.out = 51)
    # C.init.less.none <- 1 # based on cv on whole dataset
    # C.hyper.less.none <- C.init.less.none * 10^seq(-1, 1, length.out = 13)
    pred.cv.less.none <- factor(numeric(nrow(train)), levels = c(lab1, lab2))
    error.cv.less.none <- numeric(length(C.hyper.less.none))
    for (c in C.hyper.less.none) {
      for (fold2 in 1:nfolds) {
        model.cv.less.none <- lesstwoc_none(DF = train[foldid.cv != fold2, ],
                                            C = c)
        pred.cv.less.none[foldid.cv == fold2] <- predict.less_none(MODEL = model.cv.less.none,
                                                                   NEWDATA = train[foldid.cv == fold2, -ncol(train)])$prediction
      }
      error.cv.less.none[which(c == C.hyper.less.none)] <- mean(pred.cv.less.none != train[, ncol(train)])
    }
    sim.list$C.less.none[fold] <- C.hyper.less.none[
      which(error.cv.less.none == min(error.cv.less.none))[
        floor(median(1:length(which(error.cv.less.none == min(error.cv.less.none)))))]]

    # Train model
    model.less.none <- lesstwoc_none(DF = train, C = sim.list$C.less.none[fold])
    sim.list$numbetas.less.none[fold] <- sum(model.less.none$model$beta != 0)

    # Test model
    preds.labs.less.none <- factor(numeric(nrow(test)), levels = c(lab1, lab2))
    preds.labs.less.none <- factor(predict.less_none(MODEL = model.less.none,
                                                     NEWDATA = test[, -ncol(test)])$prediction,
                                   levels = c(lab1, lab2))
    score.less.none <- predict.less_none(MODEL = model.less.none,
                                         NEWDATA = test[, -ncol(test)])$score

    # Results per fold
    sim.list$auc.less.none[fold] <- pROC::roc(response = test[, ncol(test)], predictor = as.numeric(score.less.none))$auc
    sim.list$accuracy.less.none[fold] <- mean(preds.labs.less.none == test[, ncol(test)]) * 100
    sim.list$generror.less.none[fold] <- mean(preds.labs.less.none != test[, ncol(test)]) * 100



    ### LESS + standardisation ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.less.std <- 10^seq(-4, 1, length.out = 51)
    # C.init.less.std <- 0.1 # based on cv on whole dataset
    # C.hyper.less.std <- C.init.less.std * 10^seq(-1, 1, length.out = 13)
    pred.cv.less.std <- factor(numeric(nrow(train)), levels = c(lab1, lab2))
    error.cv.less.std <- numeric(length(C.hyper.less.std))
    for (c in C.hyper.less.std) {
      for (fold2 in 1:nfolds) {
        model.cv.less.std <- lesstwoc_std(DF = train[foldid.cv != fold2, ],
                                          C = c)
        pred.cv.less.std[foldid.cv == fold2] <- predict.less_std(MODEL = model.cv.less.std,
                                                                 NEWDATA = train[foldid.cv == fold2, -ncol(train)])$prediction
      }
      error.cv.less.std[which(c == C.hyper.less.std)] <- mean(pred.cv.less.std != train[, ncol(train)])
    }
    sim.list$C.less.std[fold] <- C.hyper.less.std[
      which(error.cv.less.std == min(error.cv.less.std))[
        floor(median(1:length(which(error.cv.less.std == min(error.cv.less.std)))))]]

    # Train model
    model.less.std <- lesstwoc_std(DF = train, C = sim.list$C.less.std[fold])
    sim.list$numbetas.less.std[fold] <- sum(model.less.std$model$beta != 0)

    # Test model
    preds.labs.less.std <- factor(numeric(nrow(test)), levels = c(lab1, lab2))
    preds.labs.less.std <- factor(predict.less_std(MODEL = model.less.std,
                                                   NEWDATA = test[, -ncol(test)])$prediction,
                                  levels = c(lab1, lab2))
    score.less.std <- predict.less_std(MODEL = model.less.std,
                                       NEWDATA = test[, -ncol(test)])$score

    # Results per fold
    sim.list$auc.less.std[fold] <- pROC::roc(response = test[, ncol(test)], predictor = as.numeric(score.less.std))$auc
    sim.list$accuracy.less.std[fold] <- mean(preds.labs.less.std == test[, ncol(test)]) * 100
    sim.list$generror.less.std[fold] <- mean(preds.labs.less.std != test[, ncol(test)]) * 100



    ### LESS + LESS scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.less.less <- 10^seq(-4, 1, length.out = 51)
    # C.init.less.less <- 0.5623413 # based on cv on whole dataset
    # C.hyper.less.less <- C.init.less.less * 10^seq(-1, 1, length.out = 13)
    pred.cv.less.less <- factor(numeric(nrow(train)), levels = c(lab1, lab2))
    error.cv.less.less <- numeric(length(C.hyper.less.less))
    for (c in C.hyper.less.less) {
      for (fold2 in 1:nfolds) {
        model.cv.less.less <- lesstwoc(DF = train[foldid.cv != fold2, ],
                                       C = c)
        pred.cv.less.less[foldid.cv == fold2] <- predict.less(MODEL = model.cv.less.less,
                                                              NEWDATA = train[foldid.cv == fold2, -ncol(train)])$prediction
      }
      error.cv.less.less[which(c == C.hyper.less.less)] <- mean(pred.cv.less.less != train[, ncol(train)])
    }
    sim.list$C.less.less[fold] <- C.hyper.less.less[
      which(error.cv.less.less == min(error.cv.less.less))[
        floor(median(1:length(which(error.cv.less.less == min(error.cv.less.less)))))]]

    # Train model
    model.less.less <- lesstwoc(DF = train, C = sim.list$C.less.less[fold])
    sim.list$numbetas.less.less[fold] <- sum(model.less.less$model$beta != 0)

    # Test model
    preds.labs.less.less <- factor(numeric(nrow(test)), levels = c(lab1, lab2))
    preds.labs.less.less <- factor(predict.less(MODEL = model.less.less,
                                                NEWDATA = test[, -ncol(test)])$prediction,
                                   levels = c(lab1, lab2))
    score.less.less <- predict.less(MODEL = model.less.less,
                                    NEWDATA = test[, -ncol(test)])$score

    # Results per fold
    sim.list$auc.less.less[fold] <- pROC::roc(response = test[, ncol(test)], predictor = as.numeric(score.less.less))$auc
    sim.list$accuracy.less.less[fold] <- mean(preds.labs.less.less == test[, ncol(test)]) * 100
    sim.list$generror.less.less[fold] <- mean(preds.labs.less.less != test[, ncol(test)]) * 100



    ### LESS + LESSstd scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.less.lessstd <- 10^seq(-4, 1, length.out = 51)
    # C.init.less.lessstd <- 0.3162278 # based on cv on whole dataset
    # C.hyper.less.lessstd <- C.init.less.lessstd * 10^seq(-1, 1, length.out = 13)
    pred.cv.less.lessstd <- factor(numeric(nrow(train)), levels = c(lab1, lab2))
    error.cv.less.lessstd <- numeric(length(C.hyper.less.lessstd))
    for (c in C.hyper.less.lessstd) {
      for (fold2 in 1:nfolds) {
        model.cv.less.lessstd <- lesstwoc_lessstd(DF = train[foldid.cv != fold2, ],
                                              C = c)
        pred.cv.less.lessstd[foldid.cv == fold2] <- predict.less_lessstd(MODEL = model.cv.less.lessstd,
                                                                         NEWDATA = train[foldid.cv == fold2, -ncol(train)])$prediction
      }
      error.cv.less.lessstd[which(c == C.hyper.less.lessstd)] <- mean(pred.cv.less.lessstd != train[, ncol(train)])
    }
    sim.list$C.less.lessstd[fold] <- C.hyper.less.lessstd[
      which(error.cv.less.lessstd == min(error.cv.less.lessstd))[
        floor(median(1:length(which(error.cv.less.lessstd == min(error.cv.less.lessstd)))))]]

    # Train model
    model.less.lessstd <- lesstwoc_lessstd(DF = train, C = sim.list$C.less.lessstd[fold])
    sim.list$numbetas.less.lessstd[fold] <- sum(model.less.lessstd$model$beta != 0)

    # Test model
    preds.labs.less.lessstd <- factor(numeric(nrow(test)), levels = c(lab1, lab2))
    preds.labs.less.lessstd <- factor(predict.less_lessstd(MODEL = model.less.lessstd,
                                                           NEWDATA = test[, -ncol(test)])$prediction,
                                      levels = c(lab1, lab2))
    score.less.lessstd <- predict.less_lessstd(MODEL = model.less.lessstd,
                                               NEWDATA = test[, -ncol(test)])$score

    # Results per fold
    sim.list$auc.less.lessstd[fold] <- pROC::roc(response = test[, ncol(test)], predictor = as.numeric(score.less.lessstd))$auc
    sim.list$accuracy.less.lessstd[fold] <- mean(preds.labs.less.lessstd == test[, ncol(test)]) * 100
    sim.list$generror.less.lessstd[fold] <- mean(preds.labs.less.lessstd != test[, ncol(test)]) * 100



    ### LESS + LESSstd2 scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.less.lessstd2 <- 10^seq(-4, 1, length.out = 51)
    # C.init.less.lessstd2 <- 0.03162278 # based on cv on whole dataset
    # C.hyper.less.lessstd2 <- C.init.less.lessstd2 * 10^seq(-1, 1, length.out = 13)
    pred.cv.less.lessstd2 <- factor(numeric(nrow(train)), levels = c(lab1, lab2))
    error.cv.less.lessstd2 <- numeric(length(C.hyper.less.lessstd2))
    for (c in C.hyper.less.lessstd2) {
      for (fold2 in 1:nfolds) {
        model.cv.less.lessstd2 <- lesstwoc_lessstd2(DF = train[foldid.cv != fold2, ],
                                                    C = c)
        pred.cv.less.lessstd2[foldid.cv == fold2] <- predict.less_lessstd2(MODEL = model.cv.less.lessstd2,
                                                                           NEWDATA = train[foldid.cv == fold2, -ncol(train)])$prediction
      }
      error.cv.less.lessstd2[which(c == C.hyper.less.lessstd2)] <- mean(pred.cv.less.lessstd2 != train[, ncol(train)])
    }
    sim.list$C.less.lessstd2[fold] <- C.hyper.less.lessstd2[
      which(error.cv.less.lessstd2 == min(error.cv.less.lessstd2))[
        floor(median(1:length(which(error.cv.less.lessstd2 == min(error.cv.less.lessstd2)))))]]

    # Train model
    model.less.lessstd2 <- lesstwoc_lessstd2(DF = train, C = sim.list$C.less.lessstd2[fold])
    sim.list$numbetas.less.lessstd2[fold] <- sum(model.less.lessstd2$model$beta != 0)

    # Test model
    preds.labs.less.lessstd2 <- factor(numeric(nrow(test)), levels = c(lab1, lab2))
    preds.labs.less.lessstd2 <- factor(predict.less_lessstd2(MODEL = model.less.lessstd2,
                                                             NEWDATA = test[, -ncol(test)])$prediction,
                                       levels = c(lab1, lab2))
    score.less.lessstd2 <- predict.less_lessstd2(MODEL = model.less.lessstd2,
                                                 NEWDATA = test[, -ncol(test)])$score

    # Results per fold
    sim.list$auc.less.lessstd2[fold] <- pROC::roc(response = test[, ncol(test)], predictor = as.numeric(score.less.lessstd2))$auc
    sim.list$accuracy.less.lessstd2[fold] <- mean(preds.labs.less.lessstd2 == test[, ncol(test)]) * 100
    sim.list$generror.less.lessstd2[fold] <- mean(preds.labs.less.lessstd2 != test[, ncol(test)]) * 100



    #### SVML1 #################################################################

    ### SVML1 + no scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.none <- 10^seq(-4, 1, length.out = 51)
    # C.init.svml1.none <- 1 # based on cv on whole dataset
    # C.hyper.svml1.none <- C.init.svml1.none * 10^seq(-1, 1, length.out = 13)
    pred.cv.svml1.none <- factor(numeric(nrow(train)), levels = c(lab1, lab2))
    error.cv.svml1.none <- numeric(length(C.hyper.svml1.none))
    for (c in C.hyper.svml1.none) {
      for (fold2 in 1:nfolds) {
        model.cv.svml1.none <- LiblineaR(data = train[foldid.cv != fold2, -ncol(train)],
                                         target = train[foldid.cv != fold2, ncol(train)],
                                         type = 5, #  L1-regularized L2-loss support vector classification
                                         cost = c,
                                         epsilon = 1e-7,
                                         bias = 1,
                                         wi = NULL,
                                         cross = 0,
                                         verbose = FALSE,
                                         findC = FALSE,
                                         useInitC = FALSE)
        pred.cv.svml1.none[foldid.cv == fold2] <- predict(model.cv.svml1.none,
                                                          train[foldid.cv == fold2, -ncol(train)],
                                                          decisionValues = FALSE)$predictions
      }
      error.cv.svml1.none[which(c == C.hyper.svml1.none)] <- mean(pred.cv.svml1.none != train[, ncol(train)])
    }
    sim.list$C.svml1.none <- C.hyper.svml1.none[
      which(error.cv.svml1.none == min(error.cv.svml1.none))[
        floor(median(1:length(which(error.cv.svml1.none == min(error.cv.svml1.none)))))]]

    # Train model
    model.svml1.none <- LiblineaR(data = as.matrix(train[, -ncol(train)]),
                                  target = train[, ncol(train)],
                                  type = 5, #  L1-regularized L2-loss support vector classification
                                  cost = sim.list$C.svml1.none,
                                  epsilon = 1e-7,
                                  bias = 1,
                                  wi = NULL,
                                  cross = 0,
                                  verbose = FALSE,
                                  findC = FALSE,
                                  useInitC = FALSE)
    sim.list$numbetas.svml1.none[fold] <- sum(model.svml1.none$W[-length(model.svml1.none$W)] != 0)

    # Test model
    preds.svml1.none <- predict(model.svml1.none,
                                test[, -ncol(test)],
                                decisionValues = TRUE)
    preds.labs.svml1.none <- factor(preds.svml1.none$predictions, levels = c(lab1, lab2))

    # Results per fold
    sim.list$auc.svml1.none[fold] <- pROC::roc(response = test[, ncol(test)],
                                               predictor = preds.svml1.none$decisionValues[, 1])$auc
    sim.list$accuracy.svml1.none[fold] <- mean(preds.labs.svml1.none == test[, ncol(test)]) * 100
    sim.list$generror.svml1.none[fold] <- mean(preds.labs.svml1.none != test[, ncol(test)]) * 100



    ### SVML1 + standardisation ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.std <- 10^seq(-4, 1, length.out = 51)
    # C.init.svml1.std <- 0.5623413 # based on cv on whole dataset
    # C.hyper.svml1.std <- C.init.svml1.std * 10^seq(-1, 1, length.out = 13)
    pred.cv.svml1.std <- factor(numeric(nrow(train.std)), levels = c(lab1, lab2))
    error.cv.svml1.std <- numeric(length(C.hyper.svml1.std))
    for (c in C.hyper.svml1.std) {
      for (fold2 in 1:nfolds) {
        model.cv.svml1.std <- LiblineaR(data = train.std[foldid.cv != fold2, -ncol(train.std)],
                                        target = train.std[foldid.cv != fold2, ncol(train.std)],
                                        type = 5, #  L1-regularized L2-loss support vector classification
                                        cost = c,
                                        epsilon = 1e-7,
                                        bias = 1,
                                        wi = NULL,
                                        cross = 0,
                                        verbose = FALSE,
                                        findC = FALSE,
                                        useInitC = FALSE)
        pred.cv.svml1.std[foldid.cv == fold2] <- predict(model.cv.svml1.std,
                                                         train.std[foldid.cv == fold2, -ncol(train.std)],
                                                         decisionValues = FALSE)$predictions
      }
      error.cv.svml1.std[which(c == C.hyper.svml1.std)] <- mean(pred.cv.svml1.std != train.std[, ncol(train.std)])
    }
    sim.list$C.svml1.std <- C.hyper.svml1.std[
      which(error.cv.svml1.std == min(error.cv.svml1.std))[
        floor(median(1:length(which(error.cv.svml1.std == min(error.cv.svml1.std)))))]]

    # Train model
    model.svml1.std <- LiblineaR(data = as.matrix(train.std[, -ncol(train.std)]),
                                  target = train.std[, ncol(train.std)],
                                  type = 5, #  L1-regularized L2-loss support vector classification
                                  cost = sim.list$C.svml1.std,
                                  epsilon = 1e-7,
                                  bias = 1,
                                  wi = NULL,
                                  cross = 0,
                                  verbose = FALSE,
                                  findC = FALSE,
                                  useInitC = FALSE)
    sim.list$numbetas.svml1.std[fold] <- sum(model.svml1.std$W[-length(model.svml1.std$W)] != 0)

    # Test model
    preds.svml1.std <- predict(model.svml1.std,
                               test.std[, -ncol(test.std)],
                               decisionValues = TRUE)
    preds.labs.svml1.std <- factor(preds.svml1.std$predictions, levels = c(lab1, lab2))

    # Results per fold
    sim.list$auc.svml1.std[fold] <- pROC::roc(response = test.std[, ncol(test.std)],
                                              predictor = preds.svml1.std$decisionValues[, 1])$auc
    sim.list$accuracy.svml1.std[fold] <- mean(preds.labs.svml1.std == test.std[, ncol(test.std)]) * 100
    sim.list$generror.svml1.std[fold] <- mean(preds.labs.svml1.std != test.std[, ncol(test.std)]) * 100



    ### SVML1 + LESS scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.less <- 10^seq(-4, 1, length.out = 51)
    # C.init.svml1.less <- 10 # based on cv on whole dataset
    # C.hyper.svml1.less <- C.init.svml1.less * 10^seq(-1, 1, length.out = 13)
    pred.cv.svml1.less <- factor(numeric(nrow(train.map)), levels = c(lab1, lab2))
    error.cv.svml1.less <- numeric(length(C.hyper.svml1.less))
    for (c in C.hyper.svml1.less) {
      for (fold2 in 1:nfolds) {
        model.cv.svml1.less <- LiblineaR(data = train.map[foldid.cv != fold2, -ncol(train.map)],
                                         target = train.map[foldid.cv != fold2, ncol(train.map)],
                                         type = 5, #  L1-regularized L2-loss support vector classification
                                         cost = c,
                                         epsilon = 1e-7,
                                         bias = 1,
                                         wi = NULL,
                                         cross = 0,
                                         verbose = FALSE,
                                         findC = FALSE,
                                         useInitC = FALSE)
        pred.cv.svml1.less[foldid.cv == fold2] <- predict(model.cv.svml1.less,
                                                          train.map[foldid.cv == fold2, -ncol(train.map)],
                                                          decisionValues = FALSE)$predictions
      }
      error.cv.svml1.less[which(c == C.hyper.svml1.less)] <- mean(pred.cv.svml1.less != train.map[, ncol(train.map)])
    }
    sim.list$C.svml1.less <- C.hyper.svml1.less[
      which(error.cv.svml1.less == min(error.cv.svml1.less))[
        floor(median(1:length(which(error.cv.svml1.less == min(error.cv.svml1.less)))))]]

    # Train model
    model.svml1.less <- LiblineaR(data = as.matrix(train.map[, -ncol(train.map)]),
                                  target = train.map[, ncol(train.map)],
                                  type = 5, #  L1-regularized L2-loss support vector classification
                                  cost = sim.list$C.svml1.less,
                                  epsilon = 1e-7,
                                  bias = 1,
                                  wi = NULL,
                                  cross = 0,
                                  verbose = FALSE,
                                  findC = FALSE,
                                  useInitC = FALSE)
    sim.list$numbetas.svml1.less[fold] <- sum(model.svml1.less$W[-length(model.svml1.less$W)] != 0)

    # Test model
    preds.svml1.less <- predict(model.svml1.less,
                                test.map[, -ncol(test.map)],
                                decisionValues = TRUE)
    preds.labs.svml1.less <- factor(preds.svml1.less$predictions, levels = c(lab1, lab2))

    # Results per fold
    sim.list$auc.svml1.less[fold] <- pROC::roc(response = test.map[, ncol(test.map)],
                                               predictor = preds.svml1.less$decisionValues[, 1])$auc
    sim.list$accuracy.svml1.less[fold] <- mean(preds.labs.svml1.less == test.map[, ncol(test.map)]) * 100
    sim.list$generror.svml1.less[fold] <- mean(preds.labs.svml1.less != test.map[, ncol(test.map)]) * 100



    ### SVML1 + LESSstd scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.lessstd <- 10^seq(-4, 1, length.out = 51)
    # C.init.svml1.lessstd <- 0.1778279 # based on cv on whole dataset
    # C.hyper.svml1.lessstd <- C.init.svml1.lessstd * 10^seq(-1, 1, length.out = 13)
    pred.cv.svml1.lessstd <- factor(numeric(nrow(train.map.std)), levels = c(lab1, lab2))
    error.cv.svml1.lessstd <- numeric(length(C.hyper.svml1.lessstd))
    for (c in C.hyper.svml1.lessstd) {
      for (fold2 in 1:nfolds) {
        model.cv.svml1.lessstd <- LiblineaR(data = train.map.std[foldid.cv != fold2, -ncol(train.map.std)],
                                            target = train.map.std[foldid.cv != fold2, ncol(train.map.std)],
                                            type = 5, #  L1-regularized L2-loss support vector classification
                                            cost = c,
                                            epsilon = 1e-7,
                                            bias = 1,
                                            wi = NULL,
                                            cross = 0,
                                            verbose = FALSE,
                                            findC = FALSE,
                                            useInitC = FALSE)
        pred.cv.svml1.lessstd[foldid.cv == fold2] <- predict(model.cv.svml1.lessstd,
                                                             train.map.std[foldid.cv == fold2, -ncol(train.map.std)],
                                                             decisionValues = FALSE)$predictions
      }
      error.cv.svml1.lessstd[which(c == C.hyper.svml1.lessstd)] <- mean(pred.cv.svml1.lessstd != train.map.std[, ncol(train.map.std)])
    }
    sim.list$C.svml1.lessstd <- C.hyper.svml1.lessstd[
      which(error.cv.svml1.lessstd == min(error.cv.svml1.lessstd))[
        floor(median(1:length(which(error.cv.svml1.lessstd == min(error.cv.svml1.lessstd)))))]]

    # Train model
    model.svml1.lessstd <- LiblineaR(data = as.matrix(train.map.std[, -ncol(train.map.std)]),
                                  target = train.map.std[, ncol(train.map.std)],
                                  type = 5, #  L1-regularized L2-loss support vector classification
                                  cost = sim.list$C.svml1.lessstd,
                                  epsilon = 1e-7,
                                  bias = 1,
                                  wi = NULL,
                                  cross = 0,
                                  verbose = FALSE,
                                  findC = FALSE,
                                  useInitC = FALSE)
    sim.list$numbetas.svml1.lessstd[fold] <- sum(model.svml1.lessstd$W[-length(model.svml1.lessstd$W)] != 0)

    # Test model
    preds.svml1.lessstd <- predict(model.svml1.lessstd,
                                test.map.std[, -ncol(test.map.std)],
                                decisionValues = TRUE)
    preds.labs.svml1.lessstd <- factor(preds.svml1.lessstd$predictions, levels = c(lab1, lab2))

    # Results per fold
    sim.list$auc.svml1.lessstd[fold] <- pROC::roc(response = test.map.std[, ncol(test.map.std)],
                                                  predictor = preds.svml1.lessstd$decisionValues[, 1])$auc
    sim.list$accuracy.svml1.lessstd[fold] <- mean(preds.labs.svml1.lessstd == test.map.std[, ncol(test.map.std)]) * 100
    sim.list$generror.svml1.lessstd[fold] <- mean(preds.labs.svml1.lessstd != test.map.std[, ncol(test.map.std)]) * 100



    ### SVML1 + LESSstd2 scaling ###

    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.lessstd2 <- 10^seq(-4, 1, length.out = 51)
    # C.init.svml1.lessstd2 <- 1 # based on cv on whole dataset
    # C.hyper.svml1.lessstd2 <- C.init.svml1.lessstd2 * 10^seq(-1, 1, length.out = 13)
    pred.cv.svml1.lessstd2 <- factor(numeric(nrow(train.map.std2)), levels = c(lab1, lab2))
    error.cv.svml1.lessstd2 <- numeric(length(C.hyper.svml1.lessstd2))
    for (c in C.hyper.svml1.lessstd2) {
      for (fold2 in 1:nfolds) {
        model.cv.svml1.lessstd2 <- LiblineaR(data = train.map.std2[foldid.cv != fold2, -ncol(train.map.std2)],
                                             target = train.map.std2[foldid.cv != fold2, ncol(train.map.std2)],
                                             type = 5, #  L1-regularized L2-loss support vector classification
                                             cost = c,
                                             epsilon = 1e-7,
                                             bias = 1,
                                             wi = NULL,
                                             cross = 0,
                                             verbose = FALSE,
                                             findC = FALSE,
                                             useInitC = FALSE)
        pred.cv.svml1.lessstd2[foldid.cv == fold2] <- predict(model.cv.svml1.lessstd2,
                                                              train.map.std2[foldid.cv == fold2, -ncol(train.map.std2)],
                                                              decisionValues = FALSE)$predictions
      }
      error.cv.svml1.lessstd2[which(c == C.hyper.svml1.lessstd2)] <- mean(pred.cv.svml1.lessstd2 != train.map.std2[, ncol(train.map.std2)])
    }
    sim.list$C.svml1.lessstd2 <- C.hyper.svml1.lessstd2[
      which(error.cv.svml1.lessstd2 == min(error.cv.svml1.lessstd2))[
        floor(median(1:length(which(error.cv.svml1.lessstd2 == min(error.cv.svml1.lessstd2)))))]]

    # Train model
    model.svml1.lessstd2 <- LiblineaR(data = as.matrix(train.map.std2[, -ncol(train.map.std2)]),
                                      target = train.map.std2[, ncol(train.map.std2)],
                                      type = 5, #  L1-regularized L2-loss support vector classification
                                      cost = sim.list$C.svml1.lessstd2,
                                      epsilon = 1e-7,
                                      bias = 1,
                                      wi = NULL,
                                      cross = 0,
                                      verbose = FALSE,
                                      findC = FALSE,
                                      useInitC = FALSE)
    sim.list$numbetas.svml1.lessstd2[fold] <- sum(model.svml1.lessstd2$W[-length(model.svml1.lessstd2$W)] != 0)

    # Test model
    preds.svml1.lessstd2 <- predict(model.svml1.lessstd2,
                                    test.map.std2[, -ncol(test.map.std2)],
                                    decisionValues = TRUE)
    preds.labs.svml1.lessstd2 <- factor(preds.svml1.lessstd2$predictions, levels = c(lab1, lab2))

    # Results per fold
    sim.list$auc.svml1.lessstd2[fold] <- pROC::roc(response = test.map.std2[, ncol(test.map.std2)],
                                                   predictor = preds.svml1.lessstd2$decisionValues[, 1])$auc
    sim.list$accuracy.svml1.lessstd2[fold] <- mean(preds.labs.svml1.lessstd2 == test.map.std2[, ncol(test.map.std2)]) * 100
    sim.list$generror.svml1.lessstd2[fold] <- mean(preds.labs.svml1.lessstd2 != test.map.std2[, ncol(test.map.std2)]) * 100



    #### Logistic Regression with L1 penalisation ##############################

    ### LRL1 + no scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lrl1.none <- cv.glmnet(x = as.matrix(train[, -ncol(train)]),
                                    y = train[, ncol(train)],
                                    family = "binomial",
                                    alpha = 1,
                                    nfolds = nfolds,
                                    foldid = foldid.cv,
                                    type.measure = "class",
                                    grouped = FALSE)

    # Train model
    model.lrl1.none <- glmnet(x = as.matrix(train[, -ncol(train)]),
                              y = train[, ncol(train)],
                              intercept = TRUE,
                              standardize = FALSE,
                              family = "binomial",
                              alpha = 1,
                              lambda = cv.model.lrl1.none$lambda.1se)
    sim.list$numbetas.lrl1.none[fold] <- sum(coef(model.lrl1.none)[-1] != 0)

    # Test model
    preds.lrl1.none <- predict.glmnet(object = model.lrl1.none,
                                      newx = as.matrix(test[, -ncol(test)]),
                                      s = cv.model.lrl1.none$lambda.1se,
                                      type = "class")

    # Results per fold
    preds.labs.lrl1.none <- factor(numeric(nrow(test)), levels = c(lab1, lab2))
    preds.labs.lrl1.none <- factor(ifelse(preds.lrl1.none < 0, lab1, lab2), levels = c(lab1, lab2))
    sim.list$auc.lrl1.none[fold] <- pROC::roc(response = test[, ncol(test)],
                                              predictor = as.numeric(preds.lrl1.none))$auc
    sim.list$accuracy.lrl1.none[fold] <- mean(preds.labs.lrl1.none == test[, ncol(test)]) * 100
    sim.list$generror.lrl1.none[fold] <- mean(preds.labs.lrl1.none != test[, ncol(test)]) * 100



    ### LRL1 + standardisation ###

    # 10 fold cross-validation for penalisation parameters lambda
    cv.model.lrl1.std <- cv.glmnet(x = as.matrix(train.std[, -ncol(train.std)]),
                                   y = train.std[, ncol(train.std)],
                                   family = "binomial",
                                   alpha = 1,
                                   nfolds = nfolds,
                                   foldid = foldid.cv,
                                   type.measure = "class",
                                   grouped = FALSE)

    # Train model
    model.lrl1.std <- glmnet(x = as.matrix(train.std[, -ncol(train.std)]),
                             y = train.std[, ncol(train.std)],
                             intercept = TRUE,
                             standardize = FALSE,
                             family = "binomial",
                             alpha = 1,
                             lambda = cv.model.lrl1.std$lambda.1se)
    sim.list$numbetas.lrl1.std[fold] <- sum(coef(model.lrl1.std)[-1] != 0)

    # Test model
    preds.lrl1.std <- predict.glmnet(object = model.lrl1.std,
                                     newx = as.matrix(test.std[, -ncol(test.std)]),
                                     s = cv.model.lrl1.std$lambda.1se,
                                     type = "class")

    # Results per fold
    preds.labs.lrl1.std <- factor(numeric(nrow(test.std)), levels = c(lab1, lab2))
    preds.labs.lrl1.std <- factor(ifelse(preds.lrl1.std < 0, lab1, lab2), levels = c(lab1, lab2))
    sim.list$auc.lrl1.std[fold] <- pROC::roc(response = test.std[, ncol(test.std)],
                                             predictor = as.numeric(preds.lrl1.std))$auc
    sim.list$accuracy.lrl1.std[fold] <- mean(preds.labs.lrl1.std == test.std[, ncol(test.std)]) * 100
    sim.list$generror.lrl1.std[fold] <- mean(preds.labs.lrl1.std != test.std[, ncol(test.std)]) * 100



    ### LRL1 + LESS scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lrl1.less <- cv.glmnet(x = as.matrix(train.map[, -ncol(train.map)]),
                                    y = train.map[, ncol(train.map)],
                                    family = "binomial",
                                    alpha = 1,
                                    nfolds = nfolds,
                                    foldid = foldid.cv,
                                    type.measure = "class",
                                    grouped = FALSE)

    # Train model
    model.lrl1.less <- glmnet(x = as.matrix(train.map[, -ncol(train.map)]),
                              y = train.map[, ncol(train.map)],
                              intercept = TRUE,
                              standardize = FALSE,
                              family = "binomial",
                              alpha = 1,
                              lambda = cv.model.lrl1.less$lambda.1se)
    sim.list$numbetas.lrl1.less[fold] <- sum(coef(model.lrl1.less)[-1] != 0)

    # Test model
    preds.lrl1.less <- predict.glmnet(object = model.lrl1.less,
                                      newx = as.matrix(test.map[, -ncol(test.map)]),
                                      s = cv.model.lrl1.less$lambda.1se,
                                      type = "class")

    # Results per fold
    preds.labs.lrl1.less <- factor(numeric(nrow(test.map)), levels = c(lab1, lab2))
    preds.labs.lrl1.less <- factor(ifelse(preds.lrl1.less < 0, lab1, lab2), levels = c(lab1, lab2))
    sim.list$auc.lrl1.less[fold] <- pROC::roc(response = test.map[, ncol(test.map)],
                                              predictor = as.numeric(preds.lrl1.less))$auc
    sim.list$accuracy.lrl1.less[fold] <- mean(preds.labs.lrl1.less == test.map[, ncol(test.map)]) * 100
    sim.list$generror.lrl1.less[fold] <- mean(preds.labs.lrl1.less != test.map[, ncol(test.map)]) * 100



    ### LRL1 + LESSstd scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lrl1.lessstd <- cv.glmnet(x = as.matrix(train.map.std[, -ncol(train.map.std)]),
                                       y = train.map.std[, ncol(train.map.std)],
                                       family = "binomial",
                                       alpha = 1,
                                       nfolds = nfolds,
                                       foldid = foldid.cv,
                                       type.measure = "class",
                                       grouped = FALSE)

    # Train model
    model.lrl1.lessstd <- glmnet(x = as.matrix(train.map.std[, -ncol(train.map.std)]),
                                 y = train.map.std[, ncol(train.map.std)],
                                 intercept = TRUE,
                                 standardize = FALSE,
                                 family = "binomial",
                                 alpha = 1,
                                 lambda = cv.model.lrl1.lessstd$lambda.1se)
    sim.list$numbetas.lrl1.lessstd[fold] <- sum(coef(model.lrl1.lessstd)[-1] != 0)

    # Test model
    preds.lrl1.lessstd <- predict.glmnet(object = model.lrl1.lessstd,
                                         newx = as.matrix(test.map.std[, -ncol(test.map.std)]),
                                         s = cv.model.lrl1.lessstd$lambda.1se,
                                         type = "class")

    # Results per fold
    preds.labs.lrl1.lessstd <- factor(numeric(nrow(test.map.std)), levels = c(lab1, lab2))
    preds.labs.lrl1.lessstd <- factor(ifelse(preds.lrl1.lessstd < 0, lab1, lab2), levels = c(lab1, lab2))
    sim.list$auc.lrl1.lessstd[fold] <- pROC::roc(response = test.map.std[, ncol(test.map.std)],
                                              predictor = as.numeric(preds.lrl1.lessstd))$auc
    sim.list$accuracy.lrl1.lessstd[fold] <- mean(preds.labs.lrl1.lessstd == test.map.std[, ncol(test.map.std)]) * 100
    sim.list$generror.lrl1.lessstd[fold] <- mean(preds.labs.lrl1.lessstd != test.map.std[, ncol(test.map.std)]) * 100



    ### LRL1 + LESSstd2 scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lrl1.lessstd2 <- cv.glmnet(x = as.matrix(train.map.std2[, -ncol(train.map.std2)]),
                                        y = train.map.std2[, ncol(train.map.std2)],
                                        family = "binomial",
                                        alpha = 1,
                                        nfolds = nfolds,
                                        foldid = foldid.cv,
                                        type.measure = "class",
                                        grouped = FALSE)

    # Train model
    model.lrl1.lessstd2 <- glmnet(x = as.matrix(train.map.std2[, -ncol(train.map.std2)]),
                                  y = train.map.std2[, ncol(train.map.std2)],
                                  intercept = TRUE,
                                  standardize = FALSE,
                                  family = "binomial",
                                  alpha = 1,
                                  lambda = cv.model.lrl1.lessstd2$lambda.1se)
    sim.list$numbetas.lrl1.lessstd2[fold] <- sum(coef(model.lrl1.lessstd2)[-1] != 0)

    # Test model
    preds.lrl1.lessstd2 <- predict.glmnet(object = model.lrl1.lessstd2,
                                          newx = as.matrix(test.map.std2[, -ncol(test.map.std2)]),
                                          s = cv.model.lrl1.lessstd2$lambda.1se,
                                          type = "class")

    # Results per fold
    preds.labs.lrl1.lessstd2 <- factor(numeric(nrow(test.map.std2)), levels = c(lab1, lab2))
    preds.labs.lrl1.lessstd2 <- factor(ifelse(preds.lrl1.lessstd2 < 0, lab1, lab2), levels = c(lab1, lab2))
    sim.list$auc.lrl1.lessstd2[fold] <- pROC::roc(response = test.map.std2[, ncol(test.map.std2)],
                                                  predictor = as.numeric(preds.lrl1.lessstd2))$auc
    sim.list$accuracy.lrl1.lessstd2[fold] <- mean(preds.labs.lrl1.lessstd2 == test.map.std2[, ncol(test.map.std2)]) * 100
    sim.list$generror.lrl1.lessstd2[fold] <- mean(preds.labs.lrl1.lessstd2 != test.map.std2[, ncol(test.map.std2)]) * 100



    #### LASSO Regression ######################################################

    ### LASSO + no scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lasso.none <- cv.glmnet(x = as.matrix(train[, -ncol(train)]),
                                     y = as.numeric(train[, ncol(train)]),
                                     family = "gaussian",
                                     alpha = 1,
                                     nfolds = nfolds,
                                     foldid = foldid.cv,
                                     type.measure = "mse",
                                     grouped = FALSE)

    # Train model
    model.lasso.none <- glmnet(x = as.matrix(train[, -ncol(train)]),
                               y = as.numeric(train[, ncol(train)]),
                               intercept = TRUE,
                               standardize = FALSE,
                               family = "gaussian",
                               alpha = 1,
                               lambda = cv.model.lasso.none$lambda.1se)
    sim.list$numbetas.lasso.none[fold] <- sum(coef(model.lasso.none)[-1] != 0)

    # Test model
    preds.lasso.none <- predict.glmnet(object = model.lasso.none,
                                       newx = as.matrix(test[, -ncol(test)]),
                                       s = cv.model.lasso.none$lambda.1se,
                                       type = "response")

    # Results per fold
    preds.labs.lasso.none <- factor(numeric(nrow(test)), levels = c(lab1, lab2))
    preds.labs.lasso.none <- factor(ifelse(preds.lasso.none < mean(as.numeric(unique(mydata[, ncol(mydata)]))),
                                           lab1, lab2),
                                    levels = c(lab1, lab2))
    sim.list$auc.lasso.none[fold] <- pROC::roc(response = test[, ncol(test)],
                                               predictor = as.numeric(preds.lasso.none))$auc
    sim.list$accuracy.lasso.none[fold] <- mean(preds.labs.lasso.none == test[, ncol(test)]) * 100
    sim.list$generror.lasso.none[fold] <- mean(preds.labs.lasso.none != test[, ncol(test)]) * 100



    ### LASSO + standardisation ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lasso.std <- cv.glmnet(x = as.matrix(train.std[, -ncol(train.std)]),
                                    y = as.numeric(train.std[, ncol(train.std)]),
                                    family = "gaussian",
                                    alpha = 1,
                                    nfolds = nfolds,
                                    foldid = foldid.cv,
                                    type.measure = "mse",
                                    grouped = FALSE)

    # Train model
    model.lasso.std <- glmnet(x = as.matrix(train.std[, -ncol(train.std)]),
                              y = as.numeric(train.std[, ncol(train.std)]),
                              intercept = TRUE,
                              standardize = FALSE,
                              family = "gaussian",
                              alpha = 1,
                              lambda = cv.model.lasso.std$lambda.1se)
    sim.list$numbetas.lasso.std[fold] <- sum(coef(model.lasso.std)[-1] != 0)

    # Test model
    preds.lasso.std <- predict.glmnet(object = model.lasso.std,
                                      newx = as.matrix(test.std[, -ncol(test.std)]),
                                      s = cv.model.lasso.std$lambda.1se,
                                      type = "response")

    # Results per fold
    preds.labs.lasso.std <- factor(numeric(nrow(test.std)), levels = c(lab1, lab2))
    preds.labs.lasso.std <- factor(ifelse(preds.lasso.std < mean(as.numeric(unique(mydata[, ncol(mydata)]))),
                                          lab1, lab2),
                                   levels = c(lab1, lab2))
    sim.list$auc.lasso.std[fold] <- pROC::roc(response = test.std[, ncol(test.std)],
                                              predictor = as.numeric(preds.lasso.std))$auc
    sim.list$accuracy.lasso.std[fold] <- mean(preds.labs.lasso.std == test.std[, ncol(test.std)]) * 100
    sim.list$generror.lasso.std[fold] <- mean(preds.labs.lasso.std != test.std[, ncol(test.std)]) * 100



    ### LASSO + LESS scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lasso.less <- cv.glmnet(x = as.matrix(train.map[, -ncol(train.map)]),
                                     y = as.numeric(train.map[, ncol(train.map)]),
                                     family = "gaussian",
                                     alpha = 1,
                                     nfolds = nfolds,
                                     foldid = foldid.cv,
                                     type.measure = "mse",
                                     grouped = FALSE)

    # train model
    model.lasso.less <- glmnet(x = as.matrix(train.map[, -ncol(train.map)]),
                               y = as.numeric(train.map[, ncol(train.map)]),
                               intercept = TRUE,
                               standardize = FALSE,
                               family = "gaussian",
                               alpha = 1,
                               lambda = cv.model.lasso.less$lambda.1se)
    sim.list$numbetas.lasso.less[fold] <- sum(coef(model.lasso.less)[-1] != 0)

    # Test model
    preds.lasso.less <- predict.glmnet(object = model.lasso.less,
                                       newx = as.matrix(test.map[, -ncol(test.map)]),
                                       s = cv.model.lasso.less$lambda.1se,
                                       type = "response")

    # Results per fold
    preds.labs.lasso.less <- factor(numeric(nrow(test.map)), levels = c(lab1, lab2))
    preds.labs.lasso.less <- factor(ifelse(preds.lasso.less < mean(as.numeric(unique(mydata[, ncol(mydata)]))),
                                           lab1, lab2),
                                    levels = c(lab1, lab2))
    sim.list$auc.lasso.less[fold] <- pROC::roc(response = test.map[, ncol(test.map)],
                                               predictor = as.numeric(preds.lasso.less))$auc
    sim.list$accuracy.lasso.less[fold] <- mean(preds.labs.lasso.less == test.map[, ncol(test.map)]) * 100
    sim.list$generror.lasso.less[fold] <- mean(preds.labs.lasso.less != test.map[, ncol(test.map)]) * 100



    ### LASSO + LESSstd scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lasso.lessstd <- cv.glmnet(x = as.matrix(train.map.std[, -ncol(train.map.std)]),
                                        y = as.numeric(train.map.std[, ncol(train.map.std)]),
                                        family = "gaussian",
                                        alpha = 1,
                                        nfolds = nfolds,
                                        foldid = foldid.cv,
                                        type.measure = "mse",
                                        grouped = FALSE)

    # train model
    model.lasso.lessstd <- glmnet(x = as.matrix(train.map.std[, -ncol(train.map.std)]),
                                  y = as.numeric(train.map.std[, ncol(train.map.std)]),
                                  intercept = TRUE,
                                  standardize = FALSE,
                                  family = "gaussian",
                                  alpha = 1,
                                  lambda = cv.model.lasso.lessstd$lambda.1se)
    sim.list$numbetas.lasso.lessstd[fold] <- sum(coef(model.lasso.lessstd)[-1] != 0)

    # Test model
    preds.lasso.lessstd <- predict.glmnet(object = model.lasso.lessstd,
                                          newx = as.matrix(test.map.std[, -ncol(test.map.std)]),
                                          s = cv.model.lasso.lessstd$lambda.1se,
                                          type = "response")

    # Results per fold
    preds.labs.lasso.lessstd <- factor(numeric(nrow(test.map.std)), levels = c(lab1, lab2))
    preds.labs.lasso.lessstd <- factor(ifelse(preds.lasso.lessstd < mean(as.numeric(unique(mydata[, ncol(mydata)]))),
                                              lab1, lab2),
                                       levels = c(lab1, lab2))
    sim.list$auc.lasso.lessstd[fold] <- pROC::roc(response = test.map.std[, ncol(test.map.std)],
                                                  predictor = as.numeric(preds.lasso.lessstd))$auc
    sim.list$accuracy.lasso.lessstd[fold] <- mean(preds.labs.lasso.lessstd == test.map.std[, ncol(test.map.std)]) * 100
    sim.list$generror.lasso.lessstd[fold] <- mean(preds.labs.lasso.lessstd != test.map.std[, ncol(test.map.std)]) * 100



    ### LASSO + LESSstd2 scaling ###

    # 10 fold cross-validation for penalisation parameter lambda
    cv.model.lasso.lessstd2 <- cv.glmnet(x = as.matrix(train.map.std2[, -ncol(train.map.std2)]),
                                         y = as.numeric(train.map.std2[, ncol(train.map.std2)]),
                                         family = "gaussian",
                                         alpha = 1,
                                         nfolds = nfolds,
                                         foldid = foldid.cv,
                                         type.measure = "mse",
                                         grouped = FALSE)

    # train model
    model.lasso.lessstd2 <- glmnet(x = as.matrix(train.map.std2[, -ncol(train.map.std2)]),
                                   y = as.numeric(train.map.std2[, ncol(train.map.std2)]),
                                   intercept = TRUE,
                                   standardize = FALSE,
                                   family = "gaussian",
                                   alpha = 1,
                                   lambda = cv.model.lasso.lessstd2$lambda.1se)
    sim.list$numbetas.lasso.lessstd2[fold] <- sum(coef(model.lasso.lessstd2)[-1] != 0)

    # Test model
    preds.lasso.lessstd2 <- predict.glmnet(object = model.lasso.lessstd2,
                                           newx = as.matrix(test.map.std2[, -ncol(test.map.std2)]),
                                           s = cv.model.lasso.lessstd2$lambda.1se,
                                           type = "response")

    # Results per fold
    preds.labs.lasso.lessstd2 <- factor(numeric(nrow(test.map.std2)), levels = c(lab1, lab2))
    preds.labs.lasso.lessstd2 <- factor(ifelse(preds.lasso.lessstd2 < mean(as.numeric(unique(mydata[, ncol(mydata)]))),
                                               lab1, lab2),
                                        levels = c(lab1, lab2))
    sim.list$auc.lasso.lessstd2[fold] <- pROC::roc(response = test.map.std2[, ncol(test.map.std2)],
                                                   predictor = as.numeric(preds.lasso.lessstd2))$auc
    sim.list$accuracy.lasso.lessstd2[fold] <- mean(preds.labs.lasso.lessstd2 == test.map.std2[, ncol(test.map.std2)]) * 100
    sim.list$generror.lasso.lessstd2[fold] <- mean(preds.labs.lasso.lessstd2 != test.map.std2[, ncol(test.map.std2)]) * 100
  }
  sim.list
}

stopImplicitCluster()

save(sim.list, file = "leukaemia_simlist.RData")


#### Extract results per fold per repetition [nfolds * K] ####
# Number of non-zero betas
numbetas.less.none <- unlist(lapply("numbetas.less.none", function(k) {sapply(sim.list, "[[", k)}))
numbetas.less.std <- unlist(lapply("numbetas.less.std", function(k) {sapply(sim.list, "[[", k)}))
numbetas.less.less <- unlist(lapply("numbetas.less.less", function(k) {sapply(sim.list, "[[", k)}))
numbetas.less.lessstd <- unlist(lapply("numbetas.less.lessstd", function(k) {sapply(sim.list, "[[", k)}))
numbetas.less.lessstd2 <- unlist(lapply("numbetas.less.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

numbetas.svml1.none <- unlist(lapply("numbetas.svml1.none", function(k) {sapply(sim.list, "[[", k)}))
numbetas.svml1.std <- unlist(lapply("numbetas.svml1.std", function(k) {sapply(sim.list, "[[", k)}))
numbetas.svml1.less <- unlist(lapply("numbetas.svml1.less", function(k) {sapply(sim.list, "[[", k)}))
numbetas.svml1.lessstd <- unlist(lapply("numbetas.svml1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
numbetas.svml1.lessstd2 <- unlist(lapply("numbetas.svml1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

numbetas.lrl1.none <- unlist(lapply("numbetas.lrl1.none", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lrl1.std <- unlist(lapply("numbetas.lrl1.std", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lrl1.less <- unlist(lapply("numbetas.lrl1.less", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lrl1.lessstd <- unlist(lapply("numbetas.lrl1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lrl1.lessstd2 <- unlist(lapply("numbetas.lrl1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

numbetas.lasso.none <- unlist(lapply("numbetas.lasso.none", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lasso.std <- unlist(lapply("numbetas.lasso.std", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lasso.less <- unlist(lapply("numbetas.lasso.less", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lasso.lessstd <- unlist(lapply("numbetas.lasso.lessstd", function(k) {sapply(sim.list, "[[", k)}))
numbetas.lasso.lessstd2 <- unlist(lapply("numbetas.lasso.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

# AUC
auc.less.none <- unlist(lapply("auc.less.none", function(k) {sapply(sim.list, "[[", k)}))
auc.less.std <- unlist(lapply("auc.less.std", function(k) {sapply(sim.list, "[[", k)}))
auc.less.less <- unlist(lapply("auc.less.less", function(k) {sapply(sim.list, "[[", k)}))
auc.less.lessstd <- unlist(lapply("auc.less.lessstd", function(k) {sapply(sim.list, "[[", k)}))
auc.less.lessstd2 <- unlist(lapply("auc.less.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

auc.svml1.none <- unlist(lapply("auc.svml1.none", function(k) {sapply(sim.list, "[[", k)}))
auc.svml1.std <- unlist(lapply("auc.svml1.std", function(k) {sapply(sim.list, "[[", k)}))
auc.svml1.less <- unlist(lapply("auc.svml1.less", function(k) {sapply(sim.list, "[[", k)}))
auc.svml1.lessstd <- unlist(lapply("auc.svml1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
auc.svml1.lessstd2 <- unlist(lapply("auc.svml1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

auc.lrl1.none <- unlist(lapply("auc.lrl1.none", function(k) {sapply(sim.list, "[[", k)}))
auc.lrl1.std <- unlist(lapply("auc.lrl1.std", function(k) {sapply(sim.list, "[[", k)}))
auc.lrl1.less <- unlist(lapply("auc.lrl1.less", function(k) {sapply(sim.list, "[[", k)}))
auc.lrl1.lessstd <- unlist(lapply("auc.lrl1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
auc.lrl1.lessstd2 <- unlist(lapply("auc.lrl1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

auc.lasso.none <- unlist(lapply("auc.lasso.none", function(k) {sapply(sim.list, "[[", k)}))
auc.lasso.std <- unlist(lapply("auc.lasso.std", function(k) {sapply(sim.list, "[[", k)}))
auc.lasso.less <- unlist(lapply("auc.lasso.less", function(k) {sapply(sim.list, "[[", k)}))
auc.lasso.lessstd <- unlist(lapply("auc.lasso.lessstd", function(k) {sapply(sim.list, "[[", k)}))
auc.lasso.lessstd2 <- unlist(lapply("auc.lasso.lessstd2", function(k) {sapply(sim.list, "[[", k)}))


# Accuracy
accuracy.less.none <- unlist(lapply("accuracy.less.none", function(k) {sapply(sim.list, "[[", k)}))
accuracy.less.std <- unlist(lapply("accuracy.less.std", function(k) {sapply(sim.list, "[[", k)}))
accuracy.less.less <- unlist(lapply("accuracy.less.less", function(k) {sapply(sim.list, "[[", k)}))
accuracy.less.lessstd <- unlist(lapply("accuracy.less.lessstd", function(k) {sapply(sim.list, "[[", k)}))
accuracy.less.lessstd2 <- unlist(lapply("accuracy.less.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

accuracy.svml1.none <- unlist(lapply("accuracy.svml1.none", function(k) {sapply(sim.list, "[[", k)}))
accuracy.svml1.std <- unlist(lapply("accuracy.svml1.std", function(k) {sapply(sim.list, "[[", k)}))
accuracy.svml1.less <- unlist(lapply("accuracy.svml1.less", function(k) {sapply(sim.list, "[[", k)}))
accuracy.svml1.lessstd <- unlist(lapply("accuracy.svml1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
accuracy.svml1.lessstd2 <- unlist(lapply("accuracy.svml1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

accuracy.lrl1.none <- unlist(lapply("accuracy.lrl1.none", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lrl1.std <- unlist(lapply("accuracy.lrl1.std", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lrl1.less <- unlist(lapply("accuracy.lrl1.less", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lrl1.lessstd <- unlist(lapply("accuracy.lrl1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lrl1.lessstd2 <- unlist(lapply("accuracy.lrl1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

accuracy.lasso.none <- unlist(lapply("accuracy.lasso.none", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lasso.std <- unlist(lapply("accuracy.lasso.std", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lasso.less <- unlist(lapply("accuracy.lasso.less", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lasso.lessstd <- unlist(lapply("accuracy.lasso.lessstd", function(k) {sapply(sim.list, "[[", k)}))
accuracy.lasso.lessstd2 <- unlist(lapply("accuracy.lasso.lessstd2", function(k) {sapply(sim.list, "[[", k)}))


# Generalisation error
generror.less.none <- unlist(lapply("generror.less.none", function(k) {sapply(sim.list, "[[", k)}))
generror.less.std <- unlist(lapply("generror.less.std", function(k) {sapply(sim.list, "[[", k)}))
generror.less.less <- unlist(lapply("generror.less.less", function(k) {sapply(sim.list, "[[", k)}))
generror.less.lessstd <- unlist(lapply("generror.less.lessstd", function(k) {sapply(sim.list, "[[", k)}))
generror.less.lessstd2 <- unlist(lapply("generror.less.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

generror.svml1.none <- unlist(lapply("generror.svml1.none", function(k) {sapply(sim.list, "[[", k)}))
generror.svml1.std <- unlist(lapply("generror.svml1.std", function(k) {sapply(sim.list, "[[", k)}))
generror.svml1.less <- unlist(lapply("generror.svml1.less", function(k) {sapply(sim.list, "[[", k)}))
generror.svml1.lessstd <- unlist(lapply("generror.svml1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
generror.svml1.lessstd2 <- unlist(lapply("generror.svml1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

generror.lrl1.none <- unlist(lapply("generror.lrl1.none", function(k) {sapply(sim.list, "[[", k)}))
generror.lrl1.std <- unlist(lapply("generror.lrl1.std", function(k) {sapply(sim.list, "[[", k)}))
generror.lrl1.less <- unlist(lapply("generror.lrl1.less", function(k) {sapply(sim.list, "[[", k)}))
generror.lrl1.lessstd <- unlist(lapply("generror.lrl1.lessstd", function(k) {sapply(sim.list, "[[", k)}))
generror.lrl1.lessstd2 <- unlist(lapply("generror.lrl1.lessstd2", function(k) {sapply(sim.list, "[[", k)}))

generror.lasso.none <- unlist(lapply("generror.lasso.none", function(k) {sapply(sim.list, "[[", k)}))
generror.lasso.std <- unlist(lapply("generror.lasso.std", function(k) {sapply(sim.list, "[[", k)}))
generror.lasso.less <- unlist(lapply("generror.lasso.less", function(k) {sapply(sim.list, "[[", k)}))
generror.lasso.lessstd <- unlist(lapply("generror.lasso.lessstd", function(k) {sapply(sim.list, "[[", k)}))
generror.lasso.lessstd2 <- unlist(lapply("generror.lasso.lessstd2", function(k) {sapply(sim.list, "[[", k)}))


#### Summary per fold per repetition [nfolds * K] ####
results.mydata <- data.frame(Method = factor(c(rep("LASSO", 5),
                                               rep("LogReg", 5), 
                                               rep("SVM", 5), 
                                               rep("LESS", 5)),
                                             levels = c("LASSO", "LogReg", "SVM", "LESS")),
                             Scaling = factor(rep(c("none", "std", "less", "lessstd", "lessstd2"), 4), 
                                              levels = c("none", "std", "less", "lessstd", "lessstd2")),
                             Classifier = c("LASSO_none", "LASSO_std", "LASSO_less", "LASSO_lessstd", "LASSO_lessstd2",
                                            "LogReg_none", "LogReg_std", "LogReg_less", "LogReg_lessstd", "LogReg_lessstd2", 
                                            "SVM_none", "SVM_std", "SVM_less", "SVM_lessstd", "SVM_lessstd2", 
                                            "LESS_none", "LESS_std", "LESS_less", "LESS_lessstd", "LESS_lessstd2"),
                             AUC.mean = c(mean(auc.lasso.none),
                                          mean(auc.lasso.std),
                                          mean(auc.lasso.less),
                                          mean(auc.lasso.lessstd),
                                          mean(auc.lasso.lessstd2),
                                          mean(auc.lrl1.none),
                                          mean(auc.lrl1.std),
                                          mean(auc.lrl1.less),
                                          mean(auc.lrl1.lessstd),
                                          mean(auc.lrl1.lessstd2),
                                          mean(auc.svml1.none),
                                          mean(auc.svml1.std),
                                          mean(auc.svml1.less),
                                          mean(auc.svml1.lessstd),
                                          mean(auc.svml1.lessstd2),
                                          mean(auc.less.none),
                                          mean(auc.less.std),
                                          mean(auc.less.less),
                                          mean(auc.less.lessstd),
                                          mean(auc.less.lessstd2)),
                             AUC.sd = c(sd(auc.lasso.none),
                                        sd(auc.lasso.std),
                                        sd(auc.lasso.less),
                                        sd(auc.lasso.lessstd),
                                        sd(auc.lasso.lessstd2),
                                        sd(auc.lrl1.none),
                                        sd(auc.lrl1.std),
                                        sd(auc.lrl1.less),
                                        sd(auc.lrl1.lessstd),
                                        sd(auc.lrl1.lessstd2),
                                        sd(auc.svml1.none),
                                        sd(auc.svml1.std),
                                        sd(auc.svml1.less),
                                        sd(auc.svml1.lessstd),
                                        sd(auc.svml1.lessstd2),
                                        sd(auc.less.none),
                                        sd(auc.less.std),
                                        sd(auc.less.less),
                                        sd(auc.less.lessstd),
                                        sd(auc.less.lessstd2)),
                             Accuracy.mean = c(mean(accuracy.lasso.none),
                                               mean(accuracy.lasso.std),
                                               mean(accuracy.lasso.less),
                                               mean(accuracy.lasso.lessstd),
                                               mean(accuracy.lasso.lessstd2),
                                               mean(accuracy.lrl1.none),
                                               mean(accuracy.lrl1.std),
                                               mean(accuracy.lrl1.less),
                                               mean(accuracy.lrl1.lessstd),
                                               mean(accuracy.lrl1.lessstd2),
                                               mean(accuracy.svml1.none),
                                               mean(accuracy.svml1.std),
                                               mean(accuracy.svml1.less),
                                               mean(accuracy.svml1.lessstd),
                                               mean(accuracy.svml1.lessstd2),
                                               mean(accuracy.less.none),
                                               mean(accuracy.less.std),
                                               mean(accuracy.less.less),
                                               mean(accuracy.less.lessstd),
                                               mean(accuracy.less.lessstd2)),
                             Accuracy.sd = c(sd(accuracy.lasso.none),
                                             sd(accuracy.lasso.std),
                                             sd(accuracy.lasso.less),
                                             sd(accuracy.lasso.lessstd),
                                             sd(accuracy.lasso.lessstd2),
                                             sd(accuracy.lrl1.none),
                                             sd(accuracy.lrl1.std),
                                             sd(accuracy.lrl1.less),
                                             sd(accuracy.lrl1.lessstd),
                                             sd(accuracy.lrl1.lessstd2),
                                             sd(accuracy.svml1.none),
                                             sd(accuracy.svml1.std),
                                             sd(accuracy.svml1.less),
                                             sd(accuracy.svml1.lessstd),
                                             sd(accuracy.svml1.lessstd2),
                                             sd(accuracy.less.none),
                                             sd(accuracy.less.std),
                                             sd(accuracy.less.less),
                                             sd(accuracy.less.lessstd),
                                             sd(accuracy.less.lessstd2)),
                             Generalisation.Error.mean = c(mean(generror.lasso.none),
                                                           mean(generror.lasso.std),
                                                           mean(generror.lasso.less),
                                                           mean(generror.lasso.lessstd),
                                                           mean(generror.lasso.lessstd2),
                                                           mean(generror.lrl1.none),
                                                           mean(generror.lrl1.std),
                                                           mean(generror.lrl1.less),
                                                           mean(generror.lrl1.lessstd),
                                                           mean(generror.lrl1.lessstd2),
                                                           mean(generror.svml1.none),
                                                           mean(generror.svml1.std),
                                                           mean(generror.svml1.less),
                                                           mean(generror.svml1.lessstd),
                                                           mean(generror.svml1.lessstd2),
                                                           mean(generror.less.none),
                                                           mean(generror.less.std),
                                                           mean(generror.less.less),
                                                           mean(generror.less.lessstd),
                                                           mean(generror.less.lessstd2)),
                             Generalisation.Error.sd = c(sd(generror.lasso.none),
                                                         sd(generror.lasso.std),
                                                         sd(generror.lasso.less),
                                                         sd(generror.lasso.lessstd),
                                                         sd(generror.lasso.lessstd2),
                                                         sd(generror.lrl1.none),
                                                         sd(generror.lrl1.std),
                                                         sd(generror.lrl1.less),
                                                         sd(generror.lrl1.lessstd),
                                                         sd(generror.lrl1.lessstd2),
                                                         sd(generror.svml1.none),
                                                         sd(generror.svml1.std),
                                                         sd(generror.svml1.less),
                                                         sd(generror.svml1.lessstd),
                                                         sd(generror.svml1.lessstd2),
                                                         sd(generror.less.none),
                                                         sd(generror.less.std),
                                                         sd(generror.less.less),
                                                         sd(generror.less.lessstd),
                                                         sd(generror.less.lessstd2)),
                             Dimensionality.mean = c(mean(numbetas.lasso.none),
                                                     mean(numbetas.lasso.std),
                                                     mean(numbetas.lasso.less),
                                                     mean(numbetas.lasso.lessstd),
                                                     mean(numbetas.lasso.lessstd2),
                                                     mean(numbetas.lrl1.none),
                                                     mean(numbetas.lrl1.std),
                                                     mean(numbetas.lrl1.less),
                                                     mean(numbetas.lrl1.lessstd),
                                                     mean(numbetas.lrl1.lessstd2),
                                                     mean(numbetas.svml1.none),
                                                     mean(numbetas.svml1.std),
                                                     mean(numbetas.svml1.less),
                                                     mean(numbetas.svml1.lessstd),
                                                     mean(numbetas.svml1.lessstd2),
                                                     mean(numbetas.less.none),
                                                     mean(numbetas.less.std),
                                                     mean(numbetas.less.less),
                                                     mean(numbetas.less.lessstd),
                                                     mean(numbetas.less.lessstd2)),
                             Dimensionality.sd = c(sd(numbetas.lasso.none),
                                                   sd(numbetas.lasso.std),
                                                   sd(numbetas.lasso.less),
                                                   sd(numbetas.lasso.lessstd),
                                                   sd(numbetas.lasso.lessstd2),
                                                   sd(numbetas.lrl1.none),
                                                   sd(numbetas.lrl1.std),
                                                   sd(numbetas.lrl1.less),
                                                   sd(numbetas.lrl1.lessstd),
                                                   sd(numbetas.lrl1.lessstd2),
                                                   sd(numbetas.svml1.none),
                                                   sd(numbetas.svml1.std),
                                                   sd(numbetas.svml1.less),
                                                   sd(numbetas.svml1.lessstd),
                                                   sd(numbetas.svml1.lessstd2),
                                                   sd(numbetas.less.none),
                                                   sd(numbetas.less.std),
                                                   sd(numbetas.less.less),
                                                   sd(numbetas.less.lessstd),
                                                   sd(numbetas.less.lessstd2)))

write.csv(results.mydata, "leukaemia.results.csv", row.names = FALSE)


#### Summary per repetition [K] ####
numbetas.reps.less.none      <- numeric(K)
numbetas.reps.less.std       <- numeric(K)
numbetas.reps.less.less      <- numeric(K)
numbetas.reps.less.lessstd   <- numeric(K)
numbetas.reps.less.lessstd2  <- numeric(K)
numbetas.reps.svml1.none     <- numeric(K)
numbetas.reps.svml1.std      <- numeric(K)
numbetas.reps.svml1.less     <- numeric(K)
numbetas.reps.svml1.lessstd  <- numeric(K)
numbetas.reps.svml1.lessstd2 <- numeric(K)
numbetas.reps.lrl1.none      <- numeric(K)
numbetas.reps.lrl1.std       <- numeric(K)
numbetas.reps.lrl1.less      <- numeric(K)
numbetas.reps.lrl1.lessstd   <- numeric(K)
numbetas.reps.lrl1.lessstd2  <- numeric(K)
numbetas.reps.lasso.none     <- numeric(K)
numbetas.reps.lasso.std      <- numeric(K)
numbetas.reps.lasso.less     <- numeric(K)
numbetas.reps.lasso.lessstd  <- numeric(K)
numbetas.reps.lasso.lessstd2 <- numeric(K)

auc.reps.less.none      <- numeric(K)
auc.reps.less.std       <- numeric(K)
auc.reps.less.less      <- numeric(K)
auc.reps.less.lessstd   <- numeric(K)
auc.reps.less.lessstd2  <- numeric(K)
auc.reps.svml1.none     <- numeric(K)
auc.reps.svml1.std      <- numeric(K)
auc.reps.svml1.less     <- numeric(K)
auc.reps.svml1.lessstd  <- numeric(K)
auc.reps.svml1.lessstd2 <- numeric(K)
auc.reps.lrl1.none      <- numeric(K)
auc.reps.lrl1.std       <- numeric(K)
auc.reps.lrl1.less      <- numeric(K)
auc.reps.lrl1.lessstd   <- numeric(K)
auc.reps.lrl1.lessstd2  <- numeric(K)
auc.reps.lasso.none     <- numeric(K)
auc.reps.lasso.std      <- numeric(K)
auc.reps.lasso.less     <- numeric(K)
auc.reps.lasso.lessstd  <- numeric(K)
auc.reps.lasso.lessstd2 <- numeric(K)

accuracy.reps.less.none      <- numeric(K)
accuracy.reps.less.std       <- numeric(K)
accuracy.reps.less.less      <- numeric(K)
accuracy.reps.less.lessstd   <- numeric(K)
accuracy.reps.less.lessstd2  <- numeric(K)
accuracy.reps.svml1.none     <- numeric(K)
accuracy.reps.svml1.std      <- numeric(K)
accuracy.reps.svml1.less     <- numeric(K)
accuracy.reps.svml1.lessstd  <- numeric(K)
accuracy.reps.svml1.lessstd2 <- numeric(K)
accuracy.reps.lrl1.none      <- numeric(K)
accuracy.reps.lrl1.std       <- numeric(K)
accuracy.reps.lrl1.less      <- numeric(K)
accuracy.reps.lrl1.lessstd   <- numeric(K)
accuracy.reps.lrl1.lessstd2  <- numeric(K)
accuracy.reps.lasso.none     <- numeric(K)
accuracy.reps.lasso.std      <- numeric(K)
accuracy.reps.lasso.less     <- numeric(K)
accuracy.reps.lasso.lessstd  <- numeric(K)
accuracy.reps.lasso.lessstd2 <- numeric(K)

generror.reps.less.none      <- numeric(K)
generror.reps.less.std       <- numeric(K)
generror.reps.less.less      <- numeric(K)
generror.reps.less.lessstd   <- numeric(K)
generror.reps.less.lessstd2  <- numeric(K)
generror.reps.svml1.none     <- numeric(K)
generror.reps.svml1.std      <- numeric(K)
generror.reps.svml1.less     <- numeric(K)
generror.reps.svml1.lessstd  <- numeric(K)
generror.reps.svml1.lessstd2 <- numeric(K)
generror.reps.lrl1.none      <- numeric(K)
generror.reps.lrl1.std       <- numeric(K)
generror.reps.lrl1.less      <- numeric(K)
generror.reps.lrl1.lessstd   <- numeric(K)
generror.reps.lrl1.lessstd2  <- numeric(K)
generror.reps.lasso.none     <- numeric(K)
generror.reps.lasso.std      <- numeric(K)
generror.reps.lasso.less     <- numeric(K)
generror.reps.lasso.lessstd  <- numeric(K)
generror.reps.lasso.lessstd2 <- numeric(K)


for (i in 1:K-1) {
  numbetas.reps.less.none[i + 1] <- mean(numbetas.less.none[1:nfolds + i * nfolds])
  numbetas.reps.less.std[i + 1] <- mean(numbetas.less.std[1:nfolds + i * nfolds])
  numbetas.reps.less.less[i + 1] <- mean(numbetas.less.less[1:nfolds + i * nfolds])
  numbetas.reps.less.lessstd[i + 1] <- mean(numbetas.less.lessstd[1:nfolds + i * nfolds])
  numbetas.reps.less.lessstd2[i + 1] <- mean(numbetas.less.lessstd2[1:nfolds + i * nfolds])
  numbetas.reps.svml1.none[i + 1] <- mean(numbetas.svml1.none[1:nfolds + i * nfolds])
  numbetas.reps.svml1.std[i + 1] <- mean(numbetas.svml1.std[1:nfolds + i * nfolds])
  numbetas.reps.svml1.less[i + 1] <- mean(numbetas.svml1.less[1:nfolds + i * nfolds])
  numbetas.reps.svml1.lessstd[i + 1] <- mean(numbetas.svml1.lessstd[1:nfolds + i * nfolds])
  numbetas.reps.svml1.lessstd2[i + 1] <- mean(numbetas.svml1.lessstd2[1:nfolds + i * nfolds])
  numbetas.reps.lrl1.none[i + 1] <- mean(numbetas.lrl1.none[1:nfolds + i * nfolds])
  numbetas.reps.lrl1.std[i + 1] <- mean(numbetas.lrl1.std[1:nfolds + i * nfolds])
  numbetas.reps.lrl1.less[i + 1] <- mean(numbetas.lrl1.less[1:nfolds + i * nfolds])
  numbetas.reps.lrl1.lessstd[i + 1] <- mean(numbetas.lrl1.lessstd[1:nfolds + i * nfolds])
  numbetas.reps.lrl1.lessstd2[i + 1] <- mean(numbetas.lrl1.lessstd2[1:nfolds + i * nfolds])
  numbetas.reps.lasso.none[i + 1] <- mean(numbetas.lasso.none[1:nfolds + i * nfolds])
  numbetas.reps.lasso.std[i + 1] <- mean(numbetas.lasso.std[1:nfolds + i * nfolds])
  numbetas.reps.lasso.less[i + 1] <- mean(numbetas.lasso.less[1:nfolds + i * nfolds])
  numbetas.reps.lasso.lessstd[i + 1] <- mean(numbetas.lasso.lessstd[1:nfolds + i * nfolds])
  numbetas.reps.lasso.lessstd2[i + 1] <- mean(numbetas.lasso.lessstd2[1:nfolds + i * nfolds])
  
  auc.reps.less.none[i + 1] <- mean(auc.less.none[1:nfolds + i * nfolds])
  auc.reps.less.std[i + 1] <- mean(auc.less.std[1:nfolds + i * nfolds])
  auc.reps.less.less[i + 1] <- mean(auc.less.less[1:nfolds + i * nfolds])
  auc.reps.less.lessstd[i + 1] <- mean(auc.less.lessstd[1:nfolds + i * nfolds])
  auc.reps.less.lessstd2[i + 1] <- mean(auc.less.lessstd2[1:nfolds + i * nfolds])
  auc.reps.svml1.none[i + 1] <- mean(auc.svml1.none[1:nfolds + i * nfolds])
  auc.reps.svml1.std[i + 1] <- mean(auc.svml1.std[1:nfolds + i * nfolds])
  auc.reps.svml1.less[i + 1] <- mean(auc.svml1.less[1:nfolds + i * nfolds])
  auc.reps.svml1.lessstd[i + 1] <- mean(auc.svml1.lessstd[1:nfolds + i * nfolds])
  auc.reps.svml1.lessstd2[i + 1] <- mean(auc.svml1.lessstd2[1:nfolds + i * nfolds])
  auc.reps.lrl1.none[i + 1] <- mean(auc.lrl1.none[1:nfolds + i * nfolds])
  auc.reps.lrl1.std[i + 1] <- mean(auc.lrl1.std[1:nfolds + i * nfolds])
  auc.reps.lrl1.less[i + 1] <- mean(auc.lrl1.less[1:nfolds + i * nfolds])
  auc.reps.lrl1.lessstd[i + 1] <- mean(auc.lrl1.lessstd[1:nfolds + i * nfolds])
  auc.reps.lrl1.lessstd2[i + 1] <- mean(auc.lrl1.lessstd2[1:nfolds + i * nfolds])
  auc.reps.lasso.none[i + 1] <- mean(auc.lasso.none[1:nfolds + i * nfolds])
  auc.reps.lasso.std[i + 1] <- mean(auc.lasso.std[1:nfolds + i * nfolds])
  auc.reps.lasso.less[i + 1] <- mean(auc.lasso.less[1:nfolds + i * nfolds])
  auc.reps.lasso.lessstd[i + 1] <- mean(auc.lasso.lessstd[1:nfolds + i * nfolds])
  auc.reps.lasso.lessstd2[i + 1] <- mean(auc.lasso.lessstd2[1:nfolds + i * nfolds])
  
  accuracy.reps.less.none[i + 1] <- mean(accuracy.less.none[1:nfolds + i * nfolds])
  accuracy.reps.less.std[i + 1] <- mean(accuracy.less.std[1:nfolds + i * nfolds])
  accuracy.reps.less.less[i + 1] <- mean(accuracy.less.less[1:nfolds + i * nfolds])
  accuracy.reps.less.lessstd[i + 1] <- mean(accuracy.less.lessstd[1:nfolds + i * nfolds])
  accuracy.reps.less.lessstd2[i + 1] <- mean(accuracy.less.lessstd2[1:nfolds + i * nfolds])
  accuracy.reps.svml1.none[i + 1] <- mean(accuracy.svml1.none[1:nfolds + i * nfolds])
  accuracy.reps.svml1.std[i + 1] <- mean(accuracy.svml1.std[1:nfolds + i * nfolds])
  accuracy.reps.svml1.less[i + 1] <- mean(accuracy.svml1.less[1:nfolds + i * nfolds])
  accuracy.reps.svml1.lessstd[i + 1] <- mean(accuracy.svml1.lessstd[1:nfolds + i * nfolds])
  accuracy.reps.svml1.lessstd2[i + 1] <- mean(accuracy.svml1.lessstd2[1:nfolds + i * nfolds])
  accuracy.reps.lrl1.none[i + 1] <- mean(accuracy.lrl1.none[1:nfolds + i * nfolds])
  accuracy.reps.lrl1.std[i + 1] <- mean(accuracy.lrl1.std[1:nfolds + i * nfolds])
  accuracy.reps.lrl1.less[i + 1] <- mean(accuracy.lrl1.less[1:nfolds + i * nfolds])
  accuracy.reps.lrl1.lessstd[i + 1] <- mean(accuracy.lrl1.lessstd[1:nfolds + i * nfolds])
  accuracy.reps.lrl1.lessstd2[i + 1] <- mean(accuracy.lrl1.lessstd2[1:nfolds + i * nfolds])
  accuracy.reps.lasso.none[i + 1] <- mean(accuracy.lasso.none[1:nfolds + i * nfolds])
  accuracy.reps.lasso.std[i + 1] <- mean(accuracy.lasso.std[1:nfolds + i * nfolds])
  accuracy.reps.lasso.less[i + 1] <- mean(accuracy.lasso.less[1:nfolds + i * nfolds])
  accuracy.reps.lasso.lessstd[i + 1] <- mean(accuracy.lasso.lessstd[1:nfolds + i * nfolds])
  accuracy.reps.lasso.lessstd2[i + 1] <- mean(accuracy.lasso.lessstd2[1:nfolds + i * nfolds])
  
  generror.reps.less.none[i + 1] <- mean(generror.less.none[1:nfolds + i * nfolds])
  generror.reps.less.std[i + 1] <- mean(generror.less.std[1:nfolds + i * nfolds])
  generror.reps.less.less[i + 1] <- mean(generror.less.less[1:nfolds + i * nfolds])
  generror.reps.less.lessstd[i + 1] <- mean(generror.less.lessstd[1:nfolds + i * nfolds])
  generror.reps.less.lessstd2[i + 1] <- mean(generror.less.lessstd2[1:nfolds + i * nfolds])
  generror.reps.svml1.none[i + 1] <- mean(generror.svml1.none[1:nfolds + i * nfolds])
  generror.reps.svml1.std[i + 1] <- mean(generror.svml1.std[1:nfolds + i * nfolds])
  generror.reps.svml1.less[i + 1] <- mean(generror.svml1.less[1:nfolds + i * nfolds])
  generror.reps.svml1.lessstd[i + 1] <- mean(generror.svml1.lessstd[1:nfolds + i * nfolds])
  generror.reps.svml1.lessstd2[i + 1] <- mean(generror.svml1.lessstd2[1:nfolds + i * nfolds])
  generror.reps.lrl1.none[i + 1] <- mean(generror.lrl1.none[1:nfolds + i * nfolds])
  generror.reps.lrl1.std[i + 1] <- mean(generror.lrl1.std[1:nfolds + i * nfolds])
  generror.reps.lrl1.less[i + 1] <- mean(generror.lrl1.less[1:nfolds + i * nfolds])
  generror.reps.lrl1.lessstd[i + 1] <- mean(generror.lrl1.lessstd[1:nfolds + i * nfolds])
  generror.reps.lrl1.lessstd2[i + 1] <- mean(generror.lrl1.lessstd2[1:nfolds + i * nfolds])
  generror.reps.lasso.none[i + 1] <- mean(generror.lasso.none[1:nfolds + i * nfolds])
  generror.reps.lasso.std[i + 1] <- mean(generror.lasso.std[1:nfolds + i * nfolds])
  generror.reps.lasso.less[i + 1] <- mean(generror.lasso.less[1:nfolds + i * nfolds])
  generror.reps.lasso.lessstd[i + 1] <- mean(generror.lasso.lessstd[1:nfolds + i * nfolds])
  generror.reps.lasso.lessstd2[i + 1] <- mean(generror.lasso.lessstd2[1:nfolds + i * nfolds])
}


results.reps.mydata <- data.frame(Method = factor(c(rep("LASSO", 5),
                                                    rep("LogReg", 5), 
                                                    rep("SVM", 5), 
                                                    rep("LESS", 5)),
                                                  levels = c("LASSO", "LogReg", "SVM", "LESS")),
                                  Scaling = factor(rep(c("none", "std", "less", "lessstd", "lessstd2"), 4), 
                                                   levels = c("none", "std", "less", "lessstd", "lessstd2")),
                                  Classifier = c("LASSO_none", "LASSO_std", "LASSO_less", "LASSO_lessstd", "LASSO_lessstd2", 
                                                 "LogReg_none", "LogReg_std", "LogReg_less", "LogReg_lessstd", "LogReg_lessstd2", 
                                                 "SVM_none", "SVM_std", "SVM_less", "SVM_lessstd", "SVM_lessstd2", 
                                                 "LESS_none", "LESS_std", "LESS_less", "LESS_lessstd", "LESS_lessstd2"),
                                  AUC.mean = c(mean(auc.reps.lasso.none),
                                               mean(auc.reps.lasso.std),
                                               mean(auc.reps.lasso.less),
                                               mean(auc.reps.lasso.lessstd),
                                               mean(auc.reps.lasso.lessstd2),
                                               mean(auc.reps.lrl1.none),
                                               mean(auc.reps.lrl1.std),
                                               mean(auc.reps.lrl1.less),
                                               mean(auc.reps.lrl1.lessstd),
                                               mean(auc.reps.lrl1.lessstd2),
                                               mean(auc.reps.svml1.none),
                                               mean(auc.reps.svml1.std),
                                               mean(auc.reps.svml1.less),
                                               mean(auc.reps.svml1.lessstd),
                                               mean(auc.reps.svml1.lessstd2),
                                               mean(auc.reps.less.none),
                                               mean(auc.reps.less.std),
                                               mean(auc.reps.less.less),
                                               mean(auc.reps.less.lessstd),
                                               mean(auc.reps.less.lessstd2)),
                                  AUC.sd = c(sd(auc.reps.lasso.none),
                                             sd(auc.reps.lasso.std),
                                             sd(auc.reps.lasso.less),
                                             sd(auc.reps.lasso.lessstd),
                                             sd(auc.reps.lasso.lessstd2),
                                             sd(auc.reps.lrl1.none),
                                             sd(auc.reps.lrl1.std),
                                             sd(auc.reps.lrl1.less),
                                             sd(auc.reps.lrl1.lessstd),
                                             sd(auc.reps.lrl1.lessstd2),
                                             sd(auc.reps.svml1.none),
                                             sd(auc.reps.svml1.std),
                                             sd(auc.reps.svml1.less),
                                             sd(auc.reps.svml1.lessstd),
                                             sd(auc.reps.svml1.lessstd2),
                                             sd(auc.reps.less.none),
                                             sd(auc.reps.less.std),
                                             sd(auc.reps.less.less),
                                             sd(auc.reps.less.lessstd),
                                             sd(auc.reps.less.lessstd2)),
                                  Accuracy.mean = c(mean(accuracy.reps.lasso.none),
                                                    mean(accuracy.reps.lasso.std),
                                                    mean(accuracy.reps.lasso.less),
                                                    mean(accuracy.reps.lasso.lessstd),
                                                    mean(accuracy.reps.lasso.lessstd2),
                                                    mean(accuracy.reps.lrl1.none),
                                                    mean(accuracy.reps.lrl1.std),
                                                    mean(accuracy.reps.lrl1.less),
                                                    mean(accuracy.reps.lrl1.lessstd),
                                                    mean(accuracy.reps.lrl1.lessstd2),
                                                    mean(accuracy.reps.svml1.none),
                                                    mean(accuracy.reps.svml1.std),
                                                    mean(accuracy.reps.svml1.less),
                                                    mean(accuracy.reps.svml1.lessstd),
                                                    mean(accuracy.reps.svml1.lessstd2),
                                                    mean(accuracy.reps.less.none),
                                                    mean(accuracy.reps.less.std),
                                                    mean(accuracy.reps.less.less),
                                                    mean(accuracy.reps.less.lessstd),
                                                    mean(accuracy.reps.less.lessstd2)),
                                  Accuracy.sd = c(sd(accuracy.reps.lasso.none),
                                                  sd(accuracy.reps.lasso.std),
                                                  sd(accuracy.reps.lasso.less),
                                                  sd(accuracy.reps.lasso.lessstd),
                                                  sd(accuracy.reps.lasso.lessstd2),
                                                  sd(accuracy.reps.lrl1.none),
                                                  sd(accuracy.reps.lrl1.std),
                                                  sd(accuracy.reps.lrl1.less),
                                                  sd(accuracy.reps.lrl1.lessstd),
                                                  sd(accuracy.reps.lrl1.lessstd2),
                                                  sd(accuracy.reps.svml1.none),
                                                  sd(accuracy.reps.svml1.std),
                                                  sd(accuracy.reps.svml1.less),
                                                  sd(accuracy.reps.svml1.lessstd),
                                                  sd(accuracy.reps.svml1.lessstd2),
                                                  sd(accuracy.reps.less.none),
                                                  sd(accuracy.reps.less.std),
                                                  sd(accuracy.reps.less.less),
                                                  sd(accuracy.reps.less.lessstd),
                                                  sd(accuracy.reps.less.lessstd2)),
                                  Generalisation.Error.mean = c(mean(generror.reps.lasso.none),
                                                                mean(generror.reps.lasso.std),
                                                                mean(generror.reps.lasso.less),
                                                                mean(generror.reps.lasso.lessstd),
                                                                mean(generror.reps.lasso.lessstd2),
                                                                mean(generror.reps.lrl1.none),
                                                                mean(generror.reps.lrl1.std),
                                                                mean(generror.reps.lrl1.less),
                                                                mean(generror.reps.lrl1.lessstd),
                                                                mean(generror.reps.lrl1.lessstd2),
                                                                mean(generror.reps.svml1.none),
                                                                mean(generror.reps.svml1.std),
                                                                mean(generror.reps.svml1.less),
                                                                mean(generror.reps.svml1.lessstd),
                                                                mean(generror.reps.svml1.lessstd2),
                                                                mean(generror.reps.less.none),
                                                                mean(generror.reps.less.std),
                                                                mean(generror.reps.less.less),
                                                                mean(generror.reps.less.lessstd),
                                                                mean(generror.reps.less.lessstd2)),
                                  Generalisation.Error.sd = c(sd(generror.reps.lasso.none),
                                                              sd(generror.reps.lasso.std),
                                                              sd(generror.reps.lasso.less),
                                                              sd(generror.reps.lasso.lessstd),
                                                              sd(generror.reps.lasso.lessstd2),
                                                              sd(generror.reps.lrl1.none),
                                                              sd(generror.reps.lrl1.std),
                                                              sd(generror.reps.lrl1.less),
                                                              sd(generror.reps.lrl1.lessstd),
                                                              sd(generror.reps.lrl1.lessstd2),
                                                              sd(generror.reps.svml1.none),
                                                              sd(generror.reps.svml1.std),
                                                              sd(generror.reps.svml1.less),
                                                              sd(generror.reps.svml1.lessstd),
                                                              sd(generror.reps.svml1.lessstd2),
                                                              sd(generror.reps.less.none),
                                                              sd(generror.reps.less.std),
                                                              sd(generror.reps.less.less),
                                                              sd(generror.reps.less.lessstd),
                                                              sd(generror.reps.less.lessstd2)),
                                  Dimensionality.mean = c(mean(numbetas.reps.lasso.none),
                                                          mean(numbetas.reps.lasso.std),
                                                          mean(numbetas.reps.lasso.less),
                                                          mean(numbetas.reps.lasso.lessstd),
                                                          mean(numbetas.reps.lasso.lessstd2),
                                                          mean(numbetas.reps.lrl1.none),
                                                          mean(numbetas.reps.lrl1.std),
                                                          mean(numbetas.reps.lrl1.less),
                                                          mean(numbetas.reps.lrl1.lessstd),
                                                          mean(numbetas.reps.lrl1.lessstd2),
                                                          mean(numbetas.reps.svml1.none),
                                                          mean(numbetas.reps.svml1.std),
                                                          mean(numbetas.reps.svml1.less),
                                                          mean(numbetas.reps.svml1.lessstd),
                                                          mean(numbetas.reps.svml1.lessstd2),
                                                          mean(numbetas.reps.less.none),
                                                          mean(numbetas.reps.less.std),
                                                          mean(numbetas.reps.less.less),
                                                          mean(numbetas.reps.less.lessstd),
                                                          mean(numbetas.reps.less.lessstd2)),
                                  Dimensionality.sd = c(sd(numbetas.reps.lasso.none),
                                                        sd(numbetas.reps.lasso.std),
                                                        sd(numbetas.reps.lasso.less),
                                                        sd(numbetas.reps.lasso.lessstd),
                                                        sd(numbetas.reps.lasso.lessstd2),
                                                        sd(numbetas.reps.lrl1.none),
                                                        sd(numbetas.reps.lrl1.std),
                                                        sd(numbetas.reps.lrl1.less),
                                                        sd(numbetas.reps.lrl1.lessstd),
                                                        sd(numbetas.reps.lrl1.lessstd2),
                                                        sd(numbetas.reps.svml1.none),
                                                        sd(numbetas.reps.svml1.std),
                                                        sd(numbetas.reps.svml1.less),
                                                        sd(numbetas.reps.svml1.lessstd),
                                                        sd(numbetas.reps.svml1.lessstd2),
                                                        sd(numbetas.reps.less.none),
                                                        sd(numbetas.reps.less.std),
                                                        sd(numbetas.reps.less.less),
                                                        sd(numbetas.reps.less.lessstd),
                                                        sd(numbetas.reps.less.lessstd2)))

write.csv(results.reps.mydata, "leukaemia.results.reps.csv", row.names = FALSE)


#### Boxplots of the K means over 10-fold cross-validation ####
results.all.mydata <- data.frame(Method = factor(c(rep("LASSO", 5 * K),
                                                   rep("LogReg", 5 * K), 
                                                   rep("SVM", 5 * K), 
                                                   rep("LESS", 5 * K)),
                                                 levels = c("LASSO", "LogReg", "SVM", "LESS")),
                                 Scaling = factor(rep(c(rep("none", K), 
                                                        rep("std", K), 
                                                        rep("less", K), 
                                                        rep("lessstd", K), 
                                                        rep("lessstd2", K)), 4), 
                                                  levels = c("none", "std", "less", "lessstd", "lessstd2")),
                                 Classifier = c(rep("LASSO_none", K), 
                                                rep("LASSO_std", K), 
                                                rep("LASSO_less", K), 
                                                rep("LASSO_lessstd", K), 
                                                rep("LASSO_lessstd2", K),
                                                rep("LogReg_none", K), 
                                                rep("LogReg_std", K), 
                                                rep("LogReg_less", K), 
                                                rep("LogReg_lessstd", K), 
                                                rep("LogReg_lessstd2", K),
                                                rep("SVM_none", K), 
                                                rep("SVM_std", K), 
                                                rep("SVM_less", K), 
                                                rep("SVM_lessstd", K), 
                                                rep("SVM_lessstd2", K),
                                                rep("LESS_none", K),
                                                rep("LESS_std", K), 
                                                rep("LESS_less", K), 
                                                rep("LESS_lessstd", K), 
                                                rep("LESS_lessstd2", K)),
                                 AUC = c(auc.reps.lasso.none,
                                         auc.reps.lasso.std,
                                         auc.reps.lasso.less,
                                         auc.reps.lasso.lessstd,
                                         auc.reps.lasso.lessstd2,
                                         auc.reps.lrl1.none,
                                         auc.reps.lrl1.std,
                                         auc.reps.lrl1.less,
                                         auc.reps.lrl1.lessstd,
                                         auc.reps.lrl1.lessstd2,
                                         auc.reps.svml1.none,
                                         auc.reps.svml1.std,
                                         auc.reps.svml1.less,
                                         auc.reps.svml1.lessstd,
                                         auc.reps.svml1.lessstd2,
                                         auc.reps.less.none,
                                         auc.reps.less.std,
                                         auc.reps.less.less,
                                         auc.reps.less.lessstd,
                                         auc.reps.less.lessstd2),
                                 Accuracy = c(accuracy.reps.lasso.none,
                                              accuracy.reps.lasso.std,
                                              accuracy.reps.lasso.less,
                                              accuracy.reps.lasso.lessstd,
                                              accuracy.reps.lasso.lessstd2,
                                              accuracy.reps.lrl1.none,
                                              accuracy.reps.lrl1.std,
                                              accuracy.reps.lrl1.less,
                                              accuracy.reps.lrl1.lessstd,
                                              accuracy.reps.lrl1.lessstd2,
                                              accuracy.reps.svml1.none,
                                              accuracy.reps.svml1.std,
                                              accuracy.reps.svml1.less,
                                              accuracy.reps.svml1.lessstd,
                                              accuracy.reps.svml1.lessstd2,
                                              accuracy.reps.less.none,
                                              accuracy.reps.less.std,
                                              accuracy.reps.less.less,
                                              accuracy.reps.less.lessstd,
                                              accuracy.reps.less.lessstd2),
                                 Generalisation.Error = c(generror.reps.lasso.none,
                                                          generror.reps.lasso.std,
                                                          generror.reps.lasso.less,
                                                          generror.reps.lasso.lessstd,
                                                          generror.reps.lasso.lessstd2,
                                                          generror.reps.lrl1.none,
                                                          generror.reps.lrl1.std,
                                                          generror.reps.lrl1.less,
                                                          generror.reps.lrl1.lessstd,
                                                          generror.reps.lrl1.lessstd2,
                                                          generror.reps.svml1.none,
                                                          generror.reps.svml1.std,
                                                          generror.reps.svml1.less,
                                                          generror.reps.svml1.lessstd,
                                                          generror.reps.svml1.lessstd2,
                                                          generror.reps.less.none,
                                                          generror.reps.less.std,
                                                          generror.reps.less.less,
                                                          generror.reps.less.lessstd,
                                                          generror.reps.less.lessstd2),
                                 Dimensionality = c(numbetas.reps.lasso.none,
                                                    numbetas.reps.lasso.std,
                                                    numbetas.reps.lasso.less,
                                                    numbetas.reps.lasso.lessstd,
                                                    numbetas.reps.lasso.lessstd2,
                                                    numbetas.reps.lrl1.none,
                                                    numbetas.reps.lrl1.std,
                                                    numbetas.reps.lrl1.less,
                                                    numbetas.reps.lrl1.lessstd,
                                                    numbetas.reps.lrl1.lessstd2,
                                                    numbetas.reps.svml1.none,
                                                    numbetas.reps.svml1.std,
                                                    numbetas.reps.svml1.less,
                                                    numbetas.reps.svml1.lessstd,
                                                    numbetas.reps.svml1.lessstd2,
                                                    numbetas.reps.less.none,
                                                    numbetas.reps.less.std,
                                                    numbetas.reps.less.less,
                                                    numbetas.reps.less.lessstd,
                                                    numbetas.reps.less.lessstd2))

# # Colors
groups <- 4
cols <- hcl(h = seq(15, 375, length = groups + 1), l = 65, c = 100)[1:groups]
# plot(1:groups, pch = 16, cex = 7, col = cols)
# plot(c(2:6, 9:13, 17:21, 24:28), pch = 16, cex = 7, col = cols[c(2:6, 9:13, 17:21, 24:28)])
# cols

# ggplot(results.all.mydata, aes(x = Scaling, y = AUC, fill = Method, alpha = Scaling)) +
#   geom_boxplot(lwd = 1) +
#   scale_fill_manual(values = cols[c(2, 3, 4, 1)]) +
#   facet_wrap(~ Method, nrow = 1) +
#   xlab("Classification method") +
#   ylab("AUC") +
#   theme(text = element_text(size = 16),
#         axis.text.x = element_blank())

# Remove LESS_none and LESS_std
results.all.mydata <- subset(results.all.mydata, 
                             !results.all.mydata$Classifier %in% c("LESS_none", "LESS_std"))
results.reps.mydata <- subset(results.reps.mydata, 
                              !results.reps.mydata$Classifier %in% c("LESS_none", "LESS_std"))

# 2-dimensional results (x = average model dimensionality, y = average AUC)
ggsave("leukaemia_2d.png",
       ggplot(results.reps.mydata, aes(x = Dimensionality.mean, y = AUC.mean)) +
         geom_point(aes(colour = Method, shape = Scaling), size = 5) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_shape_manual(values = c(1, 3, 16, 17, 15),
                            labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                  "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                  "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         xlab("Model dimensionality") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                shape = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(),
       width = 150, height = 150, units = "mm")


ggsave("leukaemia_auc.png",
       ggplot(results.all.mydata, aes(x = Scaling, y = AUC, fill = Method)) +
         geom_boxplot(lwd = 1) +
         scale_fill_manual(values = cols[c(2, 3, 4, 1)]) +
         facet_wrap(~ Method, nrow = 1) +
         xlab("Classification method") +
         ylab("AUC") +
         theme_bw() +
         theme(text = element_text(size = 16),
               axis.text.x = element_blank(),
               legend.position = "none"),
       width = 150, height = 150, units = "mm")

ggsave("leukaemia_accuracy.png",
       ggplot(results.all.mydata, aes(x = Scaling, y = Accuracy, fill = Method)) +
         geom_boxplot(lwd = 1) +
         scale_fill_manual(values = cols[c(2, 3, 4, 1)]) +
         facet_wrap(~ Method, nrow = 1) +
         xlab("Classification method") +
         ylab("Accuracy (%)") +
         theme_bw() +
         theme(text = element_text(size = 16),
               axis.text.x = element_blank(),
               legend.position = "none"),
       width = 150, height = 150, units = "mm")

ggsave("leukaemia_generror.png",
       ggplot(results.all.mydata, aes(x = Scaling, y = Generalisation.Error, fill = Method)) +
         geom_boxplot(lwd = 1) +
         scale_fill_manual(values = cols[c(2, 3, 4, 1)]) +
         facet_wrap(~ Method, nrow = 1) +
         xlab("Classification method") +
         ylab("Generalisation error (%)") +
         theme_bw() +
         theme(text = element_text(size = 16),
               axis.text.x = element_blank(),
               legend.position = "none"),
       width = 150, height = 150, units = "mm")

ggsave("leukaemia_numbetas.png",
       ggplot(results.all.mydata, aes(x = Scaling, y = Dimensionality, fill = Method)) +
         geom_boxplot(lwd = 1) +
         scale_fill_manual(values = cols[c(2, 3, 4, 1)]) +
         facet_wrap(~ Method, nrow = 1) +
         xlab("Classification method") +
         ylab("Model dimensionality") +
         theme_bw() +
         theme(text = element_text(size = 16),
               axis.text.x = element_blank(),
               legend.position = "none"),
       width = 150, height = 150, units = "mm")


#### Save objects ####
save(auc.less.none, auc.less.std, auc.less.less, auc.less.lessstd, auc.less.lessstd2,
     auc.svml1.none, auc.svml1.std, auc.svml1.less, auc.svml1.lessstd, auc.svml1.lessstd2,
     auc.lrl1.none, auc.lrl1.std, auc.lrl1.less, auc.lrl1.lessstd, auc.lrl1.lessstd2,
     auc.lasso.none, auc.lasso.std, auc.lasso.less, auc.lasso.lessstd, auc.lasso.lessstd2,
     accuracy.less.none, accuracy.less.std, accuracy.less.less, accuracy.less.lessstd, accuracy.less.lessstd2,
     accuracy.svml1.none, accuracy.svml1.std, accuracy.svml1.less, accuracy.svml1.lessstd, accuracy.svml1.lessstd2,
     accuracy.lrl1.none, accuracy.lrl1.std, accuracy.lrl1.less, accuracy.lrl1.lessstd, accuracy.lrl1.lessstd2,
     accuracy.lasso.none, accuracy.lasso.std, accuracy.lasso.less, accuracy.lasso.lessstd, accuracy.lasso.lessstd2,
     generror.less.none, generror.less.std, generror.less.less, generror.less.lessstd, generror.less.lessstd2,
     generror.svml1.none, generror.svml1.std, generror.svml1.less, generror.svml1.lessstd, generror.svml1.lessstd2,
     generror.lrl1.none, generror.lrl1.std, generror.lrl1.less, generror.lrl1.lessstd, generror.lrl1.lessstd2,
     generror.lasso.none, generror.lasso.std, generror.lasso.less, generror.lasso.lessstd, generror.lasso.lessstd2,
     numbetas.less.none, numbetas.less.std, numbetas.less.less, numbetas.less.lessstd, numbetas.less.lessstd2,
     numbetas.svml1.none, numbetas.svml1.std, numbetas.svml1.less, numbetas.svml1.lessstd, numbetas.svml1.lessstd2,
     numbetas.lrl1.none, numbetas.lrl1.std, numbetas.lrl1.less, numbetas.lrl1.lessstd, numbetas.lrl1.lessstd2,
     numbetas.lasso.none, numbetas.lasso.std, numbetas.lasso.less, numbetas.lasso.lessstd, numbetas.lasso.lessstd2,
     auc.reps.less.none, auc.reps.less.std, auc.reps.less.less, auc.reps.less.lessstd, auc.reps.less.lessstd2,
     auc.reps.svml1.none, auc.reps.svml1.std, auc.reps.svml1.less, auc.reps.svml1.lessstd, auc.reps.svml1.lessstd2,
     auc.reps.lrl1.none, auc.reps.lrl1.std, auc.reps.lrl1.less, auc.reps.lrl1.lessstd, auc.reps.lrl1.lessstd2,
     auc.reps.lasso.none, auc.reps.lasso.std, auc.reps.lasso.less, auc.reps.lasso.lessstd, auc.reps.lasso.lessstd2,
     accuracy.reps.less.none, accuracy.reps.less.std, accuracy.reps.less.less, accuracy.reps.less.lessstd, accuracy.reps.less.lessstd2,
     accuracy.reps.svml1.none, accuracy.reps.svml1.std, accuracy.reps.svml1.less, accuracy.reps.svml1.lessstd, accuracy.reps.svml1.lessstd2,
     accuracy.reps.lrl1.none, accuracy.reps.lrl1.std, accuracy.reps.lrl1.less, accuracy.reps.lrl1.lessstd, accuracy.reps.lrl1.lessstd2,
     accuracy.reps.lasso.none, accuracy.reps.lasso.std, accuracy.reps.lasso.less, accuracy.reps.lasso.lessstd, accuracy.reps.lasso.lessstd2,
     generror.reps.less.none, generror.reps.less.std, generror.reps.less.less, generror.reps.less.lessstd, generror.reps.less.lessstd2,
     generror.reps.svml1.none, generror.reps.svml1.std, generror.reps.svml1.less, generror.reps.svml1.lessstd, generror.reps.svml1.lessstd2,
     generror.reps.lrl1.none, generror.reps.lrl1.std, generror.reps.lrl1.less, generror.reps.lrl1.lessstd, generror.reps.lrl1.lessstd2,
     generror.reps.lasso.none, generror.reps.lasso.std, generror.reps.lasso.less, generror.reps.lasso.lessstd, generror.reps.lasso.lessstd2,
     numbetas.reps.less.none, numbetas.reps.less.std, numbetas.reps.less.less, numbetas.reps.less.lessstd, numbetas.reps.less.lessstd2,
     numbetas.reps.svml1.none, numbetas.reps.svml1.std, numbetas.reps.svml1.less, numbetas.reps.svml1.lessstd, numbetas.reps.svml1.lessstd2,
     numbetas.reps.lrl1.none, numbetas.reps.lrl1.std, numbetas.reps.lrl1.less, numbetas.reps.lrl1.lessstd, numbetas.reps.lrl1.lessstd2,
     numbetas.reps.lasso.none, numbetas.reps.lasso.std, numbetas.reps.lasso.less, numbetas.reps.lasso.lessstd, numbetas.reps.lasso.lessstd2,
     results.mydata, 
     results.reps.mydata,
     results.all.mydata,
     file = "leukaemia.RData")

send_telegram_photo(photo = "leukaemia_numbetas.png",
                    caption = "Boxplot of the model dimensionality for the leukaemia dataset", 
                    chat_id = "441084295",
                    bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")

send_telegram_photo(photo = "leukaemia_auc.png",
                    caption = "Boxplot of the AUC values for the leukaemia dataset", 
                    chat_id = "441084295",
                    bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")

send_telegram_message(text = "The leukaemia script is finished!",
                      chat_id = "441084295",
                      bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")

#### END ####