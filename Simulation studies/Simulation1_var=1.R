rm(list = ls())
set.seed(0952702)

#### General information ####
# n1 = n2 = 50, 
# V1 = meaningful variable
# V2-V1000 = not meaningful (noise) variables
# y = response variabe with labels {-1, +1}

# Load packages
library(Matrix)
library(foreach)
library(parallel)
library(iterators)
library(doParallel)
library(glmnet)
library(lpSolve)
library(ggplot2)
library(pROC)
library(LiblineaR)
library(telegram)
library(latex2exp)
source("Scripts/lesstwoc_mv.R")
source("Scripts/Telegram.R")

# General data info
n1 <- 50     # number of observations of class 1
n2 <- 50     # number of observations of class 2
N <- n1 + n2 # total number of observations
mu1 <- -0.5  # mean of class 1 on dimension 1
mu2 <- 0.5   # mean of class 2 on dimension 1
P.log <- round(10^seq(0, 3, length.out = 16)) # number of variables (logspace)
S <- 1       # variance of noise variables {0.1, 1, 10}
K <- 100     # number of simulations to average over
ncores <- 50 # number of parallel cpu cores
nfolds <- 10 # Number of folds for cross-validation

# Create empty data objects
train <- cbind(as.data.frame(matrix(NA, nrow = N, ncol = max(P.log))), 
               y = factor(c(rep(-1, n1), rep(1, n2)), levels = c("-1", "1")))

test <- cbind(as.data.frame(matrix(NA, nrow = 100 * N, ncol = max(P.log))), 
              y = factor(c(rep(-1, 100 * n1), rep(1, 100 * n2)), levels = c("-1", "1")))

# Create list for results
sim.list <- list(betas.less.none = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.less.std = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.less.less = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.less.lessstd = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.less.lessstd2 = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lrl1.none = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lrl1.std = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lrl1.less = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lrl1.lessstd = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lrl1.lessstd2 = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lasso.none = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lasso.std = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lasso.less = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lasso.lessstd = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.lasso.lessstd2 = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.svml1.none = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.svml1.std = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.svml1.less = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.svml1.lessstd = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 betas.svml1.lessstd2 = matrix(NA, nrow = length(P.log), ncol = max(P.log)),
                 
                 cv.less.none = numeric(length(P.log)),
                 cv.less.std = numeric(length(P.log)),
                 cv.less.less = numeric(length(P.log)),
                 cv.less.lessstd = numeric(length(P.log)),
                 cv.less.lessstd2 = numeric(length(P.log)),
                 cv.lrl1.none = numeric(length(P.log)),
                 cv.lrl1.std = numeric(length(P.log)),
                 cv.lrl1.less = numeric(length(P.log)),
                 cv.lrl1.lessstd = numeric(length(P.log)),
                 cv.lrl1.lessstd2 = numeric(length(P.log)),
                 cv.lasso.none = numeric(length(P.log)),
                 cv.lasso.std = numeric(length(P.log)),
                 cv.lasso.less = numeric(length(P.log)),
                 cv.lasso.lessstd = numeric(length(P.log)),
                 cv.lasso.lessstd2 = numeric(length(P.log)),
                 cv.svml1.none = numeric(length(P.log)),
                 cv.svml1.std = numeric(length(P.log)),
                 cv.svml1.less = numeric(length(P.log)),
                 cv.svml1.lessstd = numeric(length(P.log)),
                 cv.svml1.lessstd2 = numeric(length(P.log)),
                 
                 auc.less.none = numeric(length(P.log)),
                 auc.less.std = numeric(length(P.log)),
                 auc.less.less = numeric(length(P.log)),
                 auc.less.lessstd = numeric(length(P.log)),
                 auc.less.lessstd2 = numeric(length(P.log)),
                 auc.lrl1.none = numeric(length(P.log)),
                 auc.lrl1.std = numeric(length(P.log)),
                 auc.lrl1.less = numeric(length(P.log)),
                 auc.lrl1.lessstd = numeric(length(P.log)),
                 auc.lrl1.lessstd2 = numeric(length(P.log)),
                 auc.lasso.none = numeric(length(P.log)),
                 auc.lasso.std = numeric(length(P.log)),
                 auc.lasso.less = numeric(length(P.log)),
                 auc.lasso.lessstd = numeric(length(P.log)),
                 auc.lasso.lessstd2 = numeric(length(P.log)),
                 auc.svml1.none = numeric(length(P.log)),
                 auc.svml1.std = numeric(length(P.log)),
                 auc.svml1.less = numeric(length(P.log)),
                 auc.svml1.lessstd = numeric(length(P.log)),
                 auc.svml1.lessstd2 = numeric(length(P.log)),
                 
                 accuracy.less.none = numeric(length(P.log)),
                 accuracy.less.std = numeric(length(P.log)),
                 accuracy.less.less = numeric(length(P.log)),
                 accuracy.less.lessstd = numeric(length(P.log)),
                 accuracy.less.lessstd2 = numeric(length(P.log)),
                 accuracy.lrl1.none = numeric(length(P.log)),
                 accuracy.lrl1.std = numeric(length(P.log)),
                 accuracy.lrl1.less = numeric(length(P.log)),
                 accuracy.lrl1.lessstd = numeric(length(P.log)),
                 accuracy.lrl1.lessstd2 = numeric(length(P.log)),
                 accuracy.lasso.none = numeric(length(P.log)),
                 accuracy.lasso.std = numeric(length(P.log)),
                 accuracy.lasso.less = numeric(length(P.log)),
                 accuracy.lasso.lessstd = numeric(length(P.log)),
                 accuracy.lasso.lessstd2 = numeric(length(P.log)),
                 accuracy.svml1.none = numeric(length(P.log)),
                 accuracy.svml1.std = numeric(length(P.log)),
                 accuracy.svml1.less = numeric(length(P.log)),
                 accuracy.svml1.lessstd = numeric(length(P.log)),
                 accuracy.svml1.lessstd2 = numeric(length(P.log)),
                 
                 numbetas.less.none = integer(length(P.log)),
                 numbetas.less.std = integer(length(P.log)),
                 numbetas.less.less = integer(length(P.log)),
                 numbetas.less.lessstd = integer(length(P.log)),
                 numbetas.less.lessstd2 = integer(length(P.log)),
                 numbetas.lrl1.none = integer(length(P.log)),
                 numbetas.lrl1.std = integer(length(P.log)),
                 numbetas.lrl1.less = integer(length(P.log)),
                 numbetas.lrl1.lessstd = integer(length(P.log)),
                 numbetas.lrl1.lessstd2 = integer(length(P.log)),
                 numbetas.lasso.none = integer(length(P.log)),
                 numbetas.lasso.std = integer(length(P.log)),
                 numbetas.lasso.less = integer(length(P.log)),
                 numbetas.lasso.lessstd = integer(length(P.log)),
                 numbetas.lasso.lessstd2 = integer(length(P.log)),
                 numbetas.svml1.none = integer(length(P.log)),
                 numbetas.svml1.std = integer(length(P.log)),
                 numbetas.svml1.less = integer(length(P.log)),
                 numbetas.svml1.lessstd = integer(length(P.log)),
                 numbetas.svml1.lessstd2 = integer(length(P.log)),
                 
                 total.time = numeric(1))

# Go parallel
registerDoParallel(cores = ncores)

sim.list <- foreach(k = 1:K) %dopar% {
  # Progress
  print(paste("----------", "Dataset:", k, "----------"))
  start.time <- Sys.time()
  set.seed(0952702 + k)
  fold.id <- sample(rep(seq(nfolds), length = N))
  
  # Load packages
  library(Matrix)
  library(foreach)
  library(parallel)
  library(iterators)
  library(doParallel)
  library(glmnet)
  library(lpSolve)
  library(ggplot2)
  library(pROC)
  library(LiblineaR)
  library(telegram)
  library(latex2exp)
  source("Scripts/lesstwoc_mv.R")
  source("Scripts/Telegram.R")
  
  # Generate data
  train[1:n1, 1] <- rnorm(n1, mu1, 1)
  train[n1 + 1:n2, 1] <- rnorm(n2, mu2, 1)
  train[, 2:max(P.log)] <- rnorm((max(P.log) - 1) * N, 0, sqrt(S))
  
  test[1:(100 * n1), 1] <- rnorm(100 * n1, mu1, 1)
  test[100 * n1 + 1:(100 * n2), 1] <- rnorm(100 * n2, mu2, 1)
  test[, 2:max(P.log)] <- rnorm(100 * (max(P.log) - 1) * N, 0, sqrt(S))
  
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
  
  # Map data for less mapping
  M.train <- matrix(0.0, nrow = length(levels(train$y)), ncol = max(P.log))
  M.train[1, ] <- colMeans(train[train$y == levels(train$y)[1], 1:max(P.log)])
  M.train[2, ] <- colMeans(train[train$y == levels(train$y)[2], 1:max(P.log)])
  
  train.map <- train
  train.map[, 1:max(P.log)] <- mapmeans(DF = train[, -ncol(train)], M = M.train)
  
  test.map <- test
  test.map[, 1:max(P.log)] <- mapmeans(DF = test[, -ncol(test)], M = M.train)
  
  # Map data for lessstd mapping
  S.train <- matrix(0.0, nrow = length(levels(train$y)), ncol = max(P.log))
  S.train[1, ] <- apply(train[train$y == levels(train$y)[1], 1:max(P.log)], 2, var)
  S.train[2, ] <- apply(train[train$y == levels(train$y)[2], 1:max(P.log)], 2, var)
  
  train.map.std <- train
  train.map.std[, 1:max(P.log)] <- mapmeansstd(DF = train[, -ncol(train)], M = M.train, S = S.train)
  
  test.map.std <- test
  test.map.std[, 1:max(P.log)] <- mapmeansstd(DF = test[, -ncol(test)], M = M.train, S = S.train)
  
  # Map data for lessstd2 mapping
  S.train2 <- matrix(apply(rbind(train[train$y == levels(train$y)[1], 1:max(P.log)] - 
                                   matrix(rep(M.train[1, ], times = n1), nrow = n1, byrow = TRUE),
                                 train[train$y == levels(train$y)[2], 1:max(P.log)] - 
                                   matrix(rep(M.train[2, ], times = n1), nrow = n1, byrow = TRUE)), 
                           2, var), 
                     nrow = 1, ncol = max(P.log))
  
  train.map.std2 <- train
  train.map.std2[, 1:max(P.log)] <- mapmeansstd2(DF = train[, -ncol(train)], M = M.train, S = S.train2)
  
  test.map.std2 <- test
  test.map.std2[, 1:max(P.log)] <- mapmeansstd2(DF = test[, -ncol(test)], M = M.train, S = S.train2)
  
  for (p in 2:length(P.log)) {
    # Progress
    print(paste("Dimensionality:", P.log[p]))
    
    #### LESS ##################################################################
    
    ### LESS + no scaling ###
    # 10-fold cross-validation for C
    C.hyper.less.none <- 10^seq(-3, 2, length.out = 51)
    score.fold.less.none <- numeric(N) # score probabilities of fold
    auc.cv.less.none <- numeric(length(C.hyper.less.none)) # cross-validation results
    for (c in C.hyper.less.none) {
      for (fold in 1:nfolds) {
        model.fold.less.none <- lesstwoc_none(DF = train[fold.id != fold, c(1:P.log[p], ncol(train))],
                                              C = c)
        score.fold.less.none[fold.id == fold] <- predict.less_none(MODEL = model.fold.less.none,
                                                                   NEWDATA = train[fold.id == fold, 1:P.log[p]])$score
      }
      auc.cv.less.none[which(c == C.hyper.less.none)] <- pROC::roc(response = train$y, predictor = score.fold.less.none)$auc
    }
    sim.list$cv.less.none[p] <- C.hyper.less.none[
      which(auc.cv.less.none == max(auc.cv.less.none))[
        floor(median(1:length(which(auc.cv.less.none == max(auc.cv.less.none)))))]]
    # Train model
    model.less.none <- lesstwoc_none(DF = train[, c(1:P.log[p], ncol(train))],
                                     C = sim.list$cv.less.none[p])
    sim.list$betas.less.none[p, 1:P.log[p]] <- model.less.none$model$beta
    # Test model
    preds.less.none <- predict.less_none(MODEL = model.less.none,
                                         NEWDATA = test[, 1:P.log[p]])$prediction
    score.less.none <- predict.less_none(MODEL = model.less.none,
                                         NEWDATA = test[, 1:P.log[p]])$score
    sim.list$auc.less.none[p] <- pROC::roc(response = test$y, predictor = as.numeric(score.less.none))$auc
    sim.list$accuracy.less.none[p] <- mean(factor(preds.less.none, levels = c("-1", "1")) == test$y) * 100
    # print("Finished LESS + none")
    
    
    ### LESS + standardisation ###
    # 10-fold cross-validation for C
    C.hyper.less.std <- 10^seq(-3, 2, length.out = 51)
    score.fold.less.std <- numeric(N) # score probabilities of fold
    auc.cv.less.std <- numeric(length(C.hyper.less.std)) # cross-validation results
    for (c in C.hyper.less.std) {
      for (fold in 1:nfolds) {
        model.fold.less.std <- lesstwoc_std(DF = train[fold.id != fold, c(1:P.log[p], ncol(train))],
                                            C = c)
        score.fold.less.std[fold.id == fold] <- predict.less_std(MODEL = model.fold.less.std,
                                                                 NEWDATA = train[fold.id == fold, 1:P.log[p]])$score
      }
      auc.cv.less.std[which(c == C.hyper.less.std)] <- pROC::roc(response = train$y, predictor = score.fold.less.std)$auc
    }
    sim.list$cv.less.std[p] <- C.hyper.less.std[
      which(auc.cv.less.std == max(auc.cv.less.std))[
        floor(median(1:length(which(auc.cv.less.std == max(auc.cv.less.std)))))]]
    # Train model
    model.less.std <- lesstwoc_std(DF = train[, c(1:P.log[p], ncol(train))],
                                   C = sim.list$cv.less.std[p])
    sim.list$betas.less.std[p, 1:P.log[p]] <- model.less.std$model$beta
    # Test model
    preds.less.std <- predict.less_std(MODEL = model.less.std,
                                       NEWDATA = test[, 1:P.log[p]])$prediction
    score.less.std <- predict.less_std(MODEL = model.less.std,
                                       NEWDATA = test[, 1:P.log[p]])$score
    sim.list$auc.less.std[p] <- pROC::roc(response = test$y, predictor = as.numeric(score.less.std))$auc
    sim.list$accuracy.less.std[p] <- mean(factor(preds.less.std, levels = c("-1", "1")) == test$y) * 100
    # print("Finished LESS + std")
    
    
    ### LESS + LESS scaling ###
    # 10-fold cross-validation for C
    C.hyper.less.less <- 10^seq(-3, 2, length.out = 51)
    score.fold.less.less <- numeric(N) # score probabilities of fold
    auc.cv.less.less <- numeric(length(C.hyper.less.less)) # cross-validation results
    for (c in C.hyper.less.less) {
      for (fold in 1:nfolds) {
        model.fold.less.less <- lesstwoc(DF = train[fold.id != fold, c(1:P.log[p], ncol(train))],
                                         C = c)
        score.fold.less.less[fold.id == fold] <- predict.less(MODEL = model.fold.less.less,
                                                              NEWDATA = train[fold.id == fold, 1:P.log[p]])$score
      }
      auc.cv.less.less[which(c == C.hyper.less.less)] <- pROC::roc(response = train$y, predictor = score.fold.less.less)$auc
    }
    sim.list$cv.less.less[p] <- C.hyper.less.less[
      which(auc.cv.less.less == max(auc.cv.less.less))[
        floor(median(1:length(which(auc.cv.less.less == max(auc.cv.less.less)))))]]
    # Train model
    model.less.less <- lesstwoc(DF = train[, c(1:P.log[p], ncol(train))],
                                C = sim.list$cv.less.less[p])
    sim.list$betas.less.less[p, 1:P.log[p]] <- model.less.less$model$beta
    # Test model
    preds.less.less <- predict.less(MODEL = model.less.less,
                                    NEWDATA = test[, 1:P.log[p]])$prediction
    score.less.less <- predict.less(MODEL = model.less.less,
                                    NEWDATA = test[, 1:P.log[p]])$score
    sim.list$auc.less.less[p] <- pROC::roc(response = test$y, predictor = as.numeric(score.less.less))$auc
    sim.list$accuracy.less.less[p] <- mean(factor(preds.less.less, levels = c("-1", "1")) == test$y) * 100
    # print("Finished LESS + less")
    
    
    ### LESS + LESSstd scaling ###
    # 10-fold cross-validation for C
    C.hyper.less.lessstd <- 10^seq(-3, 2, length.out = 51)
    score.fold.less.lessstd <- numeric(N) # score probabilities of fold
    auc.cv.less.lessstd <- numeric(length(C.hyper.less.lessstd)) # cross-validation results
    for (c in C.hyper.less.lessstd) {
      for (fold in 1:nfolds) {
        model.fold.less.lessstd <- lesstwoc_lessstd(DF = train[fold.id != fold, c(1:P.log[p], ncol(train))],
                                                    C = c)
        score.fold.less.lessstd[fold.id == fold] <- predict.less_lessstd(MODEL = model.fold.less.lessstd,
                                                                         NEWDATA = train[fold.id == fold, 1:P.log[p]])$score
      }
      auc.cv.less.lessstd[which(c == C.hyper.less.lessstd)] <- pROC::roc(response = train$y, predictor = score.fold.less.lessstd)$auc
    }
    sim.list$cv.less.lessstd[p] <- C.hyper.less.lessstd[
      which(auc.cv.less.lessstd == max(auc.cv.less.lessstd))[
        floor(median(1:length(which(auc.cv.less.lessstd == max(auc.cv.less.lessstd)))))]]
    # Train model
    model.less.lessstd <- lesstwoc_lessstd(DF = train[, c(1:P.log[p], ncol(train))],
                                           C = sim.list$cv.less.lessstd[p])
    sim.list$betas.less.lessstd[p, 1:P.log[p]] <- model.less.lessstd$model$beta
    # Test model
    preds.less.lessstd <- predict.less_lessstd(MODEL = model.less.lessstd,
                                               NEWDATA = test[, 1:P.log[p]])$prediction
    score.less.lessstd <- predict.less_lessstd(MODEL = model.less.lessstd,
                                               NEWDATA = test[, 1:P.log[p]])$score
    sim.list$auc.less.lessstd[p] <- pROC::roc(response = test$y, predictor = as.numeric(score.less.lessstd))$auc
    sim.list$accuracy.less.lessstd[p] <- mean(factor(preds.less.lessstd, levels = c("-1", "1")) == test$y) * 100
    # print("Finished LESS + lessstd")
    
    
    ### LESS + LESSstd2 scaling ###
    # 10-fold cross-validation for C
    C.hyper.less.lessstd2 <- 10^seq(-3, 2, length.out = 51)
    score.fold.less.lessstd2 <- numeric(N) # score probabilities of fold
    auc.cv.less.lessstd2 <- numeric(length(C.hyper.less.lessstd2)) # cross-validation results
    for (c in C.hyper.less.lessstd2) {
      for (fold in 1:nfolds) {
        model.fold.less.lessstd2 <- lesstwoc_lessstd2(DF = train[fold.id != fold, c(1:P.log[p], ncol(train))],
                                                      C = c)
        score.fold.less.lessstd2[fold.id == fold] <- predict.less_lessstd2(MODEL = model.fold.less.lessstd2,
                                                                           NEWDATA = train[fold.id == fold, 1:P.log[p]])$score
      }
      auc.cv.less.lessstd2[which(c == C.hyper.less.lessstd2)] <- pROC::roc(response = train$y, predictor = score.fold.less.lessstd2)$auc
    }
    sim.list$cv.less.lessstd2[p] <- C.hyper.less.lessstd2[
      which(auc.cv.less.lessstd2 == max(auc.cv.less.lessstd2))[
        floor(median(1:length(which(auc.cv.less.lessstd2 == max(auc.cv.less.lessstd2)))))]]
    # Train model
    model.less.lessstd2 <- lesstwoc_lessstd2(DF = train[, c(1:P.log[p], ncol(train))],
                                             C = sim.list$cv.less.lessstd2[p])
    sim.list$betas.less.lessstd2[p, 1:P.log[p]] <- model.less.lessstd2$model$beta
    # Test model
    preds.less.lessstd2 <- predict.less_lessstd2(MODEL = model.less.lessstd2,
                                                 NEWDATA = test[, 1:P.log[p]])$prediction
    score.less.lessstd2 <- predict.less_lessstd2(MODEL = model.less.lessstd2,
                                                 NEWDATA = test[, 1:P.log[p]])$score
    sim.list$auc.less.lessstd2[p] <- pROC::roc(response = test$y, predictor = as.numeric(score.less.lessstd2))$auc
    sim.list$accuracy.less.lessstd2[p] <- mean(factor(preds.less.lessstd2, levels = c("-1", "1")) == test$y) * 100
    # print("Finished LESS + lessstd2")
    
    
    
    #### Support Vector Machine with L1 regularisation #########################
    
    ### SVML1 + no scaling ###
    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.none <- 10^seq(-3, 2, length.out = 51)
    score.fold.svml1.none <- numeric(N) # score probabilities of fold
    auc.cv.svml1.none <- numeric(length(C.hyper.svml1.none)) # cross-validation results
    for (c in C.hyper.svml1.none) {
      for (fold in 1:nfolds) {
        model.fold.svml1.none <- LiblineaR(data = train[fold.id != fold, c(1:P.log[p])],
                                           target = train[fold.id != fold, ncol(train)],
                                           type = 5, #  L1-regularized L2-loss support vector classification
                                           cost = c,
                                           epsilon = 1e-7,
                                           bias = 1,
                                           wi = NULL,
                                           cross = 0,
                                           verbose = FALSE,
                                           findC = FALSE,
                                           useInitC = FALSE)
        score.fold.svml1.none[fold.id == fold] <- predict(model.fold.svml1.none,
                                                          train[fold.id == fold, c(1:P.log[p])],
                                                          decisionValues = TRUE)$decisionValues[, 1]
      }
      auc.cv.svml1.none[which(c == C.hyper.svml1.none)] <- pROC::roc(response = train$y, predictor = score.fold.svml1.none)$auc
    }
    sim.list$cv.svml1.none[p] <- C.hyper.svml1.none[
      which(auc.cv.svml1.none == max(auc.cv.svml1.none))[
        floor(median(1:length(which(auc.cv.svml1.none == max(auc.cv.svml1.none)))))]]
    # Train model
    model.svml1.none <- LiblineaR(data = as.matrix(train[, c(1:P.log[p])]),
                                  target = train[, ncol(train)],
                                  type = 5, #  L1-regularized L2-loss support vector classification
                                  cost = sim.list$cv.svml1.none[p],
                                  epsilon = 1e-7,
                                  bias = 1,
                                  wi = NULL,
                                  cross = 0,
                                  verbose = FALSE,
                                  findC = FALSE,
                                  useInitC = FALSE)
    sim.list$betas.svml1.none[p, 1:P.log[p]] <- model.svml1.none$W[-length(model.svml1.none$W)]
    # Test model
    preds.svml1.none <- predict(model.svml1.none,
                                test[, 1:P.log[p]],
                                decisionValues = TRUE)$predictions
    score.svml1.none <- predict(model.svml1.none,
                                test[, 1:P.log[p]],
                                decisionValues = TRUE)$decisionValues[, 1]
    sim.list$auc.svml1.none[p] <- pROC::roc(response = test$y, predictor = as.numeric(score.svml1.none))$auc
    sim.list$accuracy.svml1.none[p] <- mean(factor(preds.svml1.none, levels = c("-1", "1")) == test$y) * 100
    # print("Finished SVML1 + none")
    
    
    ### SVML1 + standardisation ###
    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.std <- 10^seq(-3, 2, length.out = 51)
    score.fold.svml1.std <- numeric(N) # score probabilities of fold
    auc.cv.svml1.std <- numeric(length(C.hyper.svml1.std)) # cross-validation results
    for (c in C.hyper.svml1.std) {
      for (fold in 1:nfolds) {
        model.fold.svml1.std <- LiblineaR(data = train.std[fold.id != fold, c(1:P.log[p])],
                                          target = train.std[fold.id != fold, ncol(train.std)],
                                          type = 5, #  L1-regularized L2-loss support vector classification
                                          cost = c,
                                          epsilon = 1e-7,
                                          bias = 1,
                                          wi = NULL,
                                          cross = 0,
                                          verbose = FALSE,
                                          findC = FALSE,
                                          useInitC = FALSE)
        score.fold.svml1.std[fold.id == fold] <- predict(model.fold.svml1.std,
                                                         train.std[fold.id == fold, c(1:P.log[p])],
                                                         decisionValues = TRUE)$decisionValues[, 1]
      }
      auc.cv.svml1.std[which(c == C.hyper.svml1.std)] <- pROC::roc(response = train.std$y, predictor = score.fold.svml1.std)$auc
    }
    sim.list$cv.svml1.std[p] <- C.hyper.svml1.std[
      which(auc.cv.svml1.std == max(auc.cv.svml1.std))[
        floor(median(1:length(which(auc.cv.svml1.std == max(auc.cv.svml1.std)))))]]
    # Train model
    model.svml1.std <- LiblineaR(data = as.matrix(train.std[, c(1:P.log[p])]),
                                 target = train.std[, ncol(train.std)],
                                 type = 5, #  L1-regularized L2-loss support vector classification
                                 cost = sim.list$cv.svml1.std[p],
                                 epsilon = 1e-7,
                                 bias = 1,
                                 wi = NULL,
                                 cross = 0,
                                 verbose = FALSE,
                                 findC = FALSE,
                                 useInitC = FALSE)
    sim.list$betas.svml1.std[p, 1:P.log[p]] <- model.svml1.std$W[-length(model.svml1.std$W)]
    # Test model
    preds.svml1.std <- predict(model.svml1.std,
                               test.std[, 1:P.log[p]],
                               decisionValues = TRUE)$predictions
    score.svml1.std <- predict(model.svml1.std,
                               test.std[, 1:P.log[p]],
                               decisionValues = TRUE)$decisionValues[, 1]
    sim.list$auc.svml1.std[p] <- pROC::roc(response = test.std$y, predictor = as.numeric(score.svml1.std))$auc
    sim.list$accuracy.svml1.std[p] <- mean(factor(preds.svml1.std, levels = c("-1", "1")) == test.std$y) * 100
    # print("Finished SVML1 + std")
    
    
    ### SVML1 + LESS scaling ###
    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.less <- 10^seq(-3, 2, length.out = 51)
    score.fold.svml1.less <- numeric(N) # score probabilities of fold
    auc.cv.svml1.less <- numeric(length(C.hyper.svml1.less)) # cross-validation results
    for (c in C.hyper.svml1.less) {
      for (fold in 1:nfolds) {
        model.fold.svml1.less <- LiblineaR(data = train.map[fold.id != fold, c(1:P.log[p])],
                                           target = train.map[fold.id != fold, ncol(train)],
                                           type = 5, #  L1-regularized L2-loss support vector classification
                                           cost = c,
                                           epsilon = 1e-7,
                                           bias = 1,
                                           wi = NULL,
                                           cross = 0,
                                           verbose = FALSE,
                                           findC = FALSE,
                                           useInitC = FALSE)
        score.fold.svml1.less[fold.id == fold] <- predict(model.fold.svml1.less,
                                                          train.map[fold.id == fold, c(1:P.log[p])],
                                                          decisionValues = TRUE)$decisionValues[, 1]
      }
      auc.cv.svml1.less[which(c == C.hyper.svml1.less)] <- pROC::roc(response = train.map$y, predictor = score.fold.svml1.less)$auc
    }
    sim.list$cv.svml1.less[p] <- C.hyper.svml1.less[
      which(auc.cv.svml1.less == max(auc.cv.svml1.less))[
        floor(median(1:length(which(auc.cv.svml1.less == max(auc.cv.svml1.less)))))]]
    # Train model
    model.svml1.less <- LiblineaR(data = as.matrix(train.map[, c(1:P.log[p])]),
                                  target = train.map[, ncol(train)],
                                  type = 5, #  L1-regularized L2-loss support vector classification
                                  cost = sim.list$cv.svml1.less[p],
                                  epsilon = 1e-7,
                                  bias = 1,
                                  wi = NULL,
                                  cross = 0,
                                  verbose = FALSE,
                                  findC = FALSE,
                                  useInitC = FALSE)
    sim.list$betas.svml1.less[p, 1:P.log[p]] <- model.svml1.less$W[-length(model.svml1.less$W)]
    # Test model
    preds.svml1.less <- predict(model.svml1.less,
                                test.map[, 1:P.log[p]],
                                decisionValues = TRUE)$predictions
    score.svml1.less <- predict(model.svml1.less,
                                test.map[, 1:P.log[p]],
                                decisionValues = TRUE)$decisionValues[, 1]
    sim.list$auc.svml1.less[p] <- pROC::roc(response = test.map$y, predictor = as.numeric(score.svml1.less))$auc
    sim.list$accuracy.svml1.less[p] <- mean(factor(preds.svml1.less, levels = c("-1", "1")) == test.map$y) * 100
    # print("Finished SVML1 + less")
    
    
    ### SVML1 + LESSstd scaling ###
    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.lessstd <- 10^seq(-3, 2, length.out = 51)
    score.fold.svml1.lessstd <- numeric(N) # score probabilities of fold
    auc.cv.svml1.lessstd <- numeric(length(C.hyper.svml1.lessstd)) # cross-validation results
    for (c in C.hyper.svml1.lessstd) {
      for (fold in 1:nfolds) {
        model.fold.svml1.lessstd <- LiblineaR(data = train.map.std[fold.id != fold, c(1:P.log[p])],
                                              target = train.map.std[fold.id != fold, ncol(train)],
                                              type = 5, #  L1-regularized L2-loss support vector classification
                                              cost = c,
                                              epsilon = 1e-7,
                                              bias = 1,
                                              wi = NULL,
                                              cross = 0,
                                              verbose = FALSE,
                                              findC = FALSE,
                                              useInitC = FALSE)
        score.fold.svml1.lessstd[fold.id == fold] <- predict(model.fold.svml1.lessstd,
                                                             train.map.std[fold.id == fold, c(1:P.log[p])],
                                                             decisionValues = TRUE)$decisionValues[, 1]
      }
      auc.cv.svml1.lessstd[which(c == C.hyper.svml1.lessstd)] <- pROC::roc(response = train.map.std$y, predictor = score.fold.svml1.lessstd)$auc
    }
    sim.list$cv.svml1.lessstd[p] <- C.hyper.svml1.lessstd[
      which(auc.cv.svml1.lessstd == max(auc.cv.svml1.lessstd))[
        floor(median(1:length(which(auc.cv.svml1.lessstd == max(auc.cv.svml1.lessstd)))))]]
    # Train model
    model.svml1.lessstd <- LiblineaR(data = as.matrix(train.map.std[, c(1:P.log[p])]),
                                     target = train.map.std[, ncol(train)],
                                     type = 5, #  L1-regularized L2-loss support vector classification
                                     cost = sim.list$cv.svml1.lessstd[p],
                                     epsilon = 1e-7,
                                     bias = 1,
                                     wi = NULL,
                                     cross = 0,
                                     verbose = FALSE,
                                     findC = FALSE,
                                     useInitC = FALSE)
    sim.list$betas.svml1.lessstd[p, 1:P.log[p]] <- model.svml1.lessstd$W[-length(model.svml1.lessstd$W)]
    # Test model
    preds.svml1.lessstd <- predict(model.svml1.lessstd,
                                   test.map.std[, 1:P.log[p]],
                                   decisionValues = TRUE)$predictions
    score.svml1.lessstd <- predict(model.svml1.lessstd,
                                   test.map.std[, 1:P.log[p]],
                                   decisionValues = TRUE)$decisionValues[, 1]
    sim.list$auc.svml1.lessstd[p] <- pROC::roc(response = test.map.std$y, predictor = as.numeric(score.svml1.lessstd))$auc
    sim.list$accuracy.svml1.lessstd[p] <- mean(factor(preds.svml1.lessstd, levels = c("-1", "1")) == test.map.std$y) * 100
    # print("Finished SVML1 + lessstd")
    
    
    ### SVML1 + LESSstd2 scaling ###
    # 10 fold cross-validation for penalisation parameter C
    C.hyper.svml1.lessstd2 <- 10^seq(-3, 2, length.out = 51)
    score.fold.svml1.lessstd2 <- numeric(N) # score probabilities of fold
    auc.cv.svml1.lessstd2 <- numeric(length(C.hyper.svml1.lessstd2)) # cross-validation results
    for (c in C.hyper.svml1.lessstd2) {
      for (fold in 1:nfolds) {
        model.fold.svml1.lessstd2 <- LiblineaR(data = train.map.std2[fold.id != fold, c(1:P.log[p])],
                                               target = train.map.std2[fold.id != fold, ncol(train)],
                                               type = 5, #  L1-regularized L2-loss support vector classification
                                               cost = c,
                                               epsilon = 1e-7,
                                               bias = 1,
                                               wi = NULL,
                                               cross = 0,
                                               verbose = FALSE,
                                               findC = FALSE,
                                               useInitC = FALSE)
        score.fold.svml1.lessstd2[fold.id == fold] <- predict(model.fold.svml1.lessstd2,
                                                              train.map.std2[fold.id == fold, c(1:P.log[p])],
                                                              decisionValues = TRUE)$decisionValues[, 1]
      }
      auc.cv.svml1.lessstd2[which(c == C.hyper.svml1.lessstd2)] <- pROC::roc(response = train.map.std2$y, predictor = score.fold.svml1.lessstd2)$auc
    }
    sim.list$cv.svml1.lessstd2[p] <- C.hyper.svml1.lessstd2[
      which(auc.cv.svml1.lessstd2 == max(auc.cv.svml1.lessstd2))[
        floor(median(1:length(which(auc.cv.svml1.lessstd2 == max(auc.cv.svml1.lessstd2)))))]]
    # Train model
    model.svml1.lessstd2 <- LiblineaR(data = as.matrix(train.map.std2[, c(1:P.log[p])]),
                                     target = train.map.std2[, ncol(train)],
                                     type = 5, #  L1-regularized L2-loss support vector classification
                                     cost = sim.list$cv.svml1.lessstd2[p],
                                     epsilon = 1e-7,
                                     bias = 1,
                                     wi = NULL,
                                     cross = 0,
                                     verbose = FALSE,
                                     findC = FALSE,
                                     useInitC = FALSE)
    sim.list$betas.svml1.lessstd2[p, 1:P.log[p]] <- model.svml1.lessstd2$W[-length(model.svml1.lessstd2$W)]
    # Test model
    preds.svml1.lessstd2 <- predict(model.svml1.lessstd2,
                                   test.map.std2[, 1:P.log[p]],
                                   decisionValues = TRUE)$predictions
    score.svml1.lessstd2 <- predict(model.svml1.lessstd2,
                                   test.map.std2[, 1:P.log[p]],
                                   decisionValues = TRUE)$decisionValues[, 1]
    sim.list$auc.svml1.lessstd2[p] <- pROC::roc(response = test.map.std2$y, predictor = as.numeric(score.svml1.lessstd2))$auc
    sim.list$accuracy.svml1.lessstd2[p] <- mean(factor(preds.svml1.lessstd2, levels = c("-1", "1")) == test.map.std2$y) * 100
    # print("Finished SVML1 + lessstd2")
    
    
    
    #### Logistic Regression with L1 penalisation ##############################
    
    ### LRL1 + no scaling
    # 10-fold cross-validation for L1
    cv.model.lrl1.none <- cv.glmnet(x = as.matrix(train[, 1:P.log[p]]),
                                    y = train$y,
                                    family = "binomial",
                                    alpha = 1,
                                    foldid = fold.id,
                                    type.measure = "auc")
    sim.list$cv.lrl1.none[p] <- cv.model.lrl1.none$lambda.1se
    # Train model
    model.lrl1.none <- glmnet(x = as.matrix(train[, 1:P.log[p]]), 
                              y = train$y,
                              intercept = TRUE, 
                              standardize = FALSE,
                              family = "binomial",
                              alpha = 1, 
                              lambda = sim.list$cv.lrl1.none[p])
    sim.list$betas.lrl1.none[p, 1:P.log[p]] <- coef(model.lrl1.none)[-1]
    # Test model
    preds.lrl1.none <- predict.glmnet(object = model.lrl1.none,
                                      newx = as.matrix(test[, 1:P.log[p]]), 
                                      s = sim.list$cv.lrl1.none[p],
                                      type = "class")
    sim.list$auc.lrl1.none[p] <- pROC::roc(response = test$y, predictor = as.numeric(preds.lrl1.none))$auc
    preds.lrl1.none <- factor(ifelse(preds.lrl1.none < 0, -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lrl1.none[p] <- mean(preds.lrl1.none == test$y) * 100
    # print("Finished LRL1 + none")
    
    
    ### LRL1 + standardisation
    # 10-fold cross-validation for L1
    cv.model.lrl1.std <- cv.glmnet(x = as.matrix(train.std[, 1:P.log[p]]),
                                   y = train.std$y,
                                   family = "binomial",
                                   alpha = 1,
                                   foldid = fold.id,
                                   type.measure = "auc")
    sim.list$cv.lrl1.std[p] <- cv.model.lrl1.std$lambda.1se
    # Train model
    model.lrl1.std <- glmnet(x = as.matrix(train.std[, 1:P.log[p]]), 
                             y = train.std$y,
                             intercept = TRUE, 
                             standardize = FALSE,
                             family = "binomial",
                             alpha = 1, 
                             lambda = sim.list$cv.lrl1.std[p])
    sim.list$betas.lrl1.std[p, 1:P.log[p]] <- coef(model.lrl1.std)[-1]
    # Test model
    preds.lrl1.std <- predict.glmnet(object = model.lrl1.std,
                                     newx = as.matrix(test.std[, 1:P.log[p]]), 
                                     s = sim.list$cv.lrl1.std[p],
                                     type = "class")
    sim.list$auc.lrl1.std[p] <- pROC::roc(response = test.std$y, predictor = as.numeric(preds.lrl1.std))$auc
    preds.lrl1.std <- factor(ifelse(preds.lrl1.std < 0, -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lrl1.std[p] <- mean(preds.lrl1.std == test.std$y) * 100
    # print("Finished LRL1 + std")
    
    
    ### LRL1 + LESS scaling
    # 10-fold cross-validation for L1
    cv.model.lrl1.less <- cv.glmnet(x = as.matrix(train.map[, 1:P.log[p]]),
                                    y = train.map$y,
                                    family = "binomial",
                                    alpha = 1,
                                    foldid = fold.id,
                                    type.measure = "auc")
    sim.list$cv.lrl1.less[p] <- cv.model.lrl1.less$lambda.1se
    # Train model
    model.lrl1.less <- glmnet(x = as.matrix(train.map[, 1:P.log[p]]), 
                              y = train.map$y,
                              intercept = TRUE, 
                              standardize = FALSE,
                              family = "binomial",
                              alpha = 1, 
                              lambda = sim.list$cv.lrl1.less[p])
    sim.list$betas.lrl1.less[p, 1:P.log[p]] <- coef(model.lrl1.less)[-1]
    # Test model
    preds.lrl1.less <- predict.glmnet(object = model.lrl1.less,
                                      newx = as.matrix(test.map[, 1:P.log[p]]), 
                                      s = sim.list$cv.lrl1.less[p],
                                      type = "class")
    sim.list$auc.lrl1.less[p] <- pROC::roc(response = test.map$y, predictor = as.numeric(preds.lrl1.less))$auc
    preds.lrl1.less <- factor(ifelse(preds.lrl1.less < 0, -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lrl1.less[p] <- mean(preds.lrl1.less == test.map$y) * 100
    # print("Finished LRL1 + less")
    
    
    ### LRL1 + LESSstd scaling
    # 10-fold cross-validation for L1
    cv.model.lrl1.lessstd <- cv.glmnet(x = as.matrix(train.map.std[, 1:P.log[p]]),
                                       y = train.map.std$y,
                                       family = "binomial",
                                       alpha = 1,
                                       foldid = fold.id,
                                       type.measure = "auc")
    sim.list$cv.lrl1.lessstd[p] <- cv.model.lrl1.lessstd$lambda.1se
    # Train model
    model.lrl1.lessstd <- glmnet(x = as.matrix(train.map.std[, 1:P.log[p]]), 
                                 y = train.map.std$y,
                                 intercept = TRUE,
                                 standardize = FALSE,
                                 family = "binomial",
                                 alpha = 1,
                                 lambda = sim.list$cv.lrl1.lessstd[p])
    sim.list$betas.lrl1.lessstd[p, 1:P.log[p]] <- coef(model.lrl1.lessstd)[-1]
    # Test model
    preds.lrl1.lessstd <- predict.glmnet(object = model.lrl1.lessstd,
                                         newx = as.matrix(test.map.std[, 1:P.log[p]]),
                                         s = sim.list$cv.lrl1.lessstd[p],
                                         type = "class")
    sim.list$auc.lrl1.lessstd[p] <- pROC::roc(response = test.map.std$y, predictor = as.numeric(preds.lrl1.lessstd))$auc
    preds.lrl1.lessstd <- factor(ifelse(preds.lrl1.lessstd < 0, -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lrl1.lessstd[p] <- mean(preds.lrl1.lessstd == test.map.std$y) * 100
    # print("Finished LRL1 + lessstd")
    
    
    ### LRL1 + LESSstd scaling
    # 10-fold cross-validation for L1
    cv.model.lrl1.lessstd2 <- cv.glmnet(x = as.matrix(train.map.std2[, 1:P.log[p]]),
                                        y = train.map.std2$y,
                                        family = "binomial",
                                        alpha = 1,
                                        foldid = fold.id,
                                        type.measure = "auc")
    sim.list$cv.lrl1.lessstd2[p] <- cv.model.lrl1.lessstd2$lambda.1se
    # Train model
    model.lrl1.lessstd2 <- glmnet(x = as.matrix(train.map.std2[, 1:P.log[p]]), 
                                  y = train.map.std2$y,
                                  intercept = TRUE,
                                  standardize = FALSE,
                                  family = "binomial",
                                  alpha = 1,
                                  lambda = sim.list$cv.lrl1.lessstd2[p])
    sim.list$betas.lrl1.lessstd2[p, 1:P.log[p]] <- coef(model.lrl1.lessstd2)[-1]
    # Test model
    preds.lrl1.lessstd2 <- predict.glmnet(object = model.lrl1.lessstd2,
                                          newx = as.matrix(test.map.std2[, 1:P.log[p]]),
                                          s = sim.list$cv.lrl1.lessstd2[p],
                                          type = "class")
    sim.list$auc.lrl1.lessstd2[p] <- pROC::roc(response = test.map.std2$y, predictor = as.numeric(preds.lrl1.lessstd2))$auc
    preds.lrl1.lessstd2 <- factor(ifelse(preds.lrl1.lessstd2 < 0, -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lrl1.lessstd2[p] <- mean(preds.lrl1.lessstd2 == test.map.std2$y) * 100
    # print("Finished LRL1 + lessstd2")
    
    
    
    #### LASSO Regression ######################################################
    
    ### LASSO + no scaling
    # 10-fold cross-validation for L1
    cv.model.lasso.none <- cv.glmnet(x = as.matrix(train[, 1:P.log[p]]),
                                     y = as.numeric(train$y),
                                     family = "gaussian",
                                     alpha = 1,
                                     foldid = fold.id,
                                     type.measure = "mse")
    sim.list$cv.lasso.none[p] <- cv.model.lasso.none$lambda.1se
    # Train model
    model.lasso.none <- glmnet(x = as.matrix(train[, 1:P.log[p]]), 
                               y = as.numeric(train$y),
                               intercept = TRUE, 
                               standardize = FALSE,
                               family = "gaussian",
                               alpha = 1, 
                               lambda = sim.list$cv.lasso.none[p])
    sim.list$betas.lasso.none[p, 1:P.log[p]] <- coef(model.lasso.none)[-1]
    # Test model
    preds.lasso.none <- predict.glmnet(object = model.lasso.none,
                                       newx = as.matrix(test[, 1:P.log[p]]),
                                       s = sim.list$cv.lasso.none[p],
                                       type = "link")
    sim.list$auc.lasso.none[p] <- pROC::roc(response = test$y, predictor = as.numeric(preds.lasso.none))$auc
    preds.lasso.none <- factor(ifelse(preds.lasso.none < mean(unique(as.numeric(train$y))), -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lasso.none[p] <- mean(preds.lasso.none == test$y) * 100
    # print("Finished LASSO + none")
    
    
    ### LASSO + standardisation
    # 10-fold cross-validation for L1
    cv.model.lasso.std <- cv.glmnet(x = as.matrix(train.std[, 1:P.log[p]]),
                                    y = as.numeric(train.std$y),
                                    family = "gaussian",
                                    alpha = 1,
                                    foldid = fold.id,
                                    type.measure = "mse")
    sim.list$cv.lasso.std[p] <- cv.model.lasso.std$lambda.1se
    # Train model
    model.lasso.std <- glmnet(x = as.matrix(train.std[, 1:P.log[p]]),
                              y = as.numeric(train.std$y),
                              intercept = TRUE, 
                              standardize = FALSE,
                              family = "gaussian",
                              alpha = 1, 
                              lambda = sim.list$cv.lasso.std[p])
    sim.list$betas.lasso.std[p, 1:P.log[p]] <- coef(model.lasso.std)[-1]
    # Test model
    preds.lasso.std <- predict.glmnet(object = model.lasso.std,
                                      newx = as.matrix(test.std[, 1:P.log[p]]),
                                      s = sim.list$cv.lasso.std[p],
                                      type = "link")
    sim.list$auc.lasso.std[p] <- pROC::roc(response = test.std$y, predictor = as.numeric(preds.lasso.std))$auc
    preds.lasso.std <- factor(ifelse(preds.lasso.std < mean(unique(as.numeric(train.std$y))), -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lasso.std[p] <- mean(preds.lasso.std == test.std$y) * 100
    # print("Finished LASSO + std")
    
    
    ### LASSO + LESS scaling
    # 10-fold cross-validation for L1
    cv.model.lasso.less <- cv.glmnet(x = as.matrix(train.map[, 1:P.log[p]]),
                                     y = as.numeric(train.map$y),
                                     family = "gaussian",
                                     alpha = 1,
                                     foldid = fold.id,
                                     type.measure = "mse")
    sim.list$cv.lasso.less[p] <- cv.model.lasso.less$lambda.1se
    # Train model
    model.lasso.less <- glmnet(x = as.matrix(train.map[, 1:P.log[p]]), 
                               y = as.numeric(train.map$y),
                               intercept = TRUE, 
                               standardize = FALSE,
                               family = "gaussian",
                               alpha = 1, 
                               lambda = sim.list$cv.lasso.less[p])
    sim.list$betas.lasso.less[p, 1:P.log[p]] <- coef(model.lasso.less)[-1]
    # Test model
    preds.lasso.less <- predict.glmnet(object = model.lasso.less,
                                       newx = as.matrix(test.map[, 1:P.log[p]]),
                                       s = sim.list$cv.lasso.less[p],
                                       type = "link")
    sim.list$auc.lasso.less[p] <- pROC::roc(response = test.map$y, predictor = as.numeric(preds.lasso.less))$auc
    preds.lasso.less <- factor(ifelse(preds.lasso.less < mean(unique(as.numeric(train.map$y))), -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lasso.less[p] <- mean(preds.lasso.less == test.map$y) * 100
    # print("Finished LASSO + less")
    
    
    ### LASSO + LESSstd scaling
    # 10-fold cross-validation for L1
    cv.model.lasso.lessstd <- cv.glmnet(x = as.matrix(train.map.std[, 1:P.log[p]]),
                                        y = as.numeric(train.map.std$y),
                                        family = "gaussian",
                                        alpha = 1,
                                        foldid = fold.id,
                                        type.measure = "mse")
    sim.list$cv.lasso.lessstd[p] <- cv.model.lasso.lessstd$lambda.1se
    # Train model
    model.lasso.lessstd <- glmnet(x = as.matrix(train.map.std[, 1:P.log[p]]), 
                                  y = as.numeric(train.map.std$y),
                                  intercept = TRUE, 
                                  standardize = FALSE,
                                  family = "gaussian",
                                  alpha = 1, 
                                  lambda = sim.list$cv.lasso.lessstd[p])
    sim.list$betas.lasso.lessstd[p, 1:P.log[p]] <- coef(model.lasso.lessstd)[-1]
    # Test model
    preds.lasso.lessstd <- predict.glmnet(object = model.lasso.lessstd,
                                          newx = as.matrix(test.map.std[, 1:P.log[p]]),
                                          s = sim.list$cv.lasso.lessstd[p],
                                          type = "link")
    sim.list$auc.lasso.lessstd[p] <- pROC::roc(response = test.map.std$y, predictor = as.numeric(preds.lasso.lessstd))$auc
    preds.lasso.lessstd <- factor(ifelse(preds.lasso.lessstd < mean(unique(as.numeric(train.map.std$y))), -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lasso.lessstd[p] <- mean(preds.lasso.lessstd == test.map.std$y) * 100
    # print("Finished LASSO + lessstd")
    
    
    ### LASSO + LESSstd2 scaling
    # 10-fold cross-validation for L1
    cv.model.lasso.lessstd2 <- cv.glmnet(x = as.matrix(train.map.std2[, 1:P.log[p]]),
                                         y = as.numeric(train.map.std2$y),
                                         family = "gaussian",
                                         alpha = 1,
                                         foldid = fold.id,
                                         type.measure = "mse")
    sim.list$cv.lasso.lessstd2[p] <- cv.model.lasso.lessstd2$lambda.1se
    # Train model
    model.lasso.lessstd2 <- glmnet(x = as.matrix(train.map.std2[, 1:P.log[p]]), 
                                   y = as.numeric(train.map.std2$y),
                                   intercept = TRUE, 
                                   standardize = FALSE,
                                   family = "gaussian",
                                   alpha = 1, 
                                   lambda = sim.list$cv.lasso.lessstd2[p])
    sim.list$betas.lasso.lessstd2[p, 1:P.log[p]] <- coef(model.lasso.lessstd2)[-1]
    # Test model
    preds.lasso.lessstd2 <- predict.glmnet(object = model.lasso.lessstd2,
                                           newx = as.matrix(test.map.std2[, 1:P.log[p]]),
                                           s = sim.list$cv.lasso.lessstd2[p],
                                           type = "link")
    sim.list$auc.lasso.lessstd2[p] <- pROC::roc(response = test.map.std2$y, predictor = as.numeric(preds.lasso.lessstd2))$auc
    preds.lasso.lessstd2 <- factor(ifelse(preds.lasso.lessstd2 < mean(unique(as.numeric(train.map.std2$y))), -1, 1), levels = c("-1", "1"))
    sim.list$accuracy.lasso.lessstd2[p] <- mean(preds.lasso.lessstd2 == test.map.std2$y) * 100
    # print("Finished LASSO + lessstd2")
    
  }
  sim.list$numbetas.less.none <- apply(sim.list$betas.less.none, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.less.std <- apply(sim.list$betas.less.std, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.less.less <- apply(sim.list$betas.less.less, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.less.lessstd <- apply(sim.list$betas.less.lessstd, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.less.lessstd2 <- apply(sim.list$betas.less.lessstd2, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  
  sim.list$numbetas.svml1.none <- apply(sim.list$betas.svml1.none, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.svml1.std <- apply(sim.list$betas.svml1.std, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.svml1.less <- apply(sim.list$betas.svml1.less, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.svml1.lessstd <- apply(sim.list$betas.svml1.lessstd, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.svml1.lessstd2 <- apply(sim.list$betas.svml1.lessstd2, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  
  sim.list$numbetas.lrl1.none <- apply(sim.list$betas.lrl1.none, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lrl1.std <- apply(sim.list$betas.lrl1.std, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lrl1.less <- apply(sim.list$betas.lrl1.less, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lrl1.lessstd <- apply(sim.list$betas.lrl1.lessstd, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lrl1.lessstd2 <- apply(sim.list$betas.lrl1.lessstd2, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  
  sim.list$numbetas.lasso.none <- apply(sim.list$betas.lasso.none, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lasso.std <- apply(sim.list$betas.lasso.std, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lasso.less <- apply(sim.list$betas.lasso.less, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lasso.lessstd <- apply(sim.list$betas.lasso.lessstd, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  sim.list$numbetas.lasso.lessstd2 <- apply(sim.list$betas.lasso.lessstd2, 1, function(x) {sum(x != 0 & abs(x) > 1e-6, na.rm = TRUE)})
  
  end.time <- Sys.time()
  sim.list$total.time <- as.numeric(difftime(end.time, start.time, units = "sec"))
  sim.list
}

stopImplicitCluster()

# Total time
total.time <- unlist(lapply("total.time", function(k) {sum(sapply(sim.list, "[[", k))}))

# Mean Model Sparseness over all K simulations
sim.mean.numbetas.less.none <- unlist(lapply("numbetas.less.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.less.std <- unlist(lapply("numbetas.less.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.less.less <- unlist(lapply("numbetas.less.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.less.lessstd <- unlist(lapply("numbetas.less.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.less.lessstd2 <- unlist(lapply("numbetas.less.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.numbetas.svml1.none <- unlist(lapply("numbetas.svml1.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.svml1.std <- unlist(lapply("numbetas.svml1.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.svml1.less <- unlist(lapply("numbetas.svml1.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.svml1.lessstd <- unlist(lapply("numbetas.svml1.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.svml1.lessstd2 <- unlist(lapply("numbetas.svml1.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.numbetas.lrl1.none <- unlist(lapply("numbetas.lrl1.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lrl1.std <- unlist(lapply("numbetas.lrl1.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lrl1.less <- unlist(lapply("numbetas.lrl1.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lrl1.lessstd <- unlist(lapply("numbetas.lrl1.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lrl1.lessstd2 <- unlist(lapply("numbetas.lrl1.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.numbetas.lasso.none <- unlist(lapply("numbetas.lasso.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lasso.std <- unlist(lapply("numbetas.lasso.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lasso.less <- unlist(lapply("numbetas.lasso.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lasso.lessstd <- unlist(lapply("numbetas.lasso.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.numbetas.lasso.lessstd2 <- unlist(lapply("numbetas.lasso.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))


# Mean Test AUC over all K simulations
sim.mean.auc.less.none <- unlist(lapply("auc.less.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.less.std <- unlist(lapply("auc.less.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.less.less <- unlist(lapply("auc.less.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.less.lessstd <- unlist(lapply("auc.less.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.less.lessstd2 <- unlist(lapply("auc.less.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.auc.svml1.none <- unlist(lapply("auc.svml1.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.svml1.std <- unlist(lapply("auc.svml1.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.svml1.less <- unlist(lapply("auc.svml1.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.svml1.lessstd <- unlist(lapply("auc.svml1.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.svml1.lessstd2 <- unlist(lapply("auc.svml1.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.auc.lrl1.none <- unlist(lapply("auc.lrl1.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lrl1.std <- unlist(lapply("auc.lrl1.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lrl1.less <- unlist(lapply("auc.lrl1.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lrl1.lessstd <- unlist(lapply("auc.lrl1.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lrl1.lessstd2 <- unlist(lapply("auc.lrl1.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.auc.lasso.none <- unlist(lapply("auc.lasso.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lasso.std <- unlist(lapply("auc.lasso.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lasso.less <- unlist(lapply("auc.lasso.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lasso.lessstd <- unlist(lapply("auc.lasso.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.auc.lasso.lessstd2 <- unlist(lapply("auc.lasso.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))


# Mean Test Accuracy over all K simulations
sim.mean.accuracy.less.none <- unlist(lapply("accuracy.less.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.less.std <- unlist(lapply("accuracy.less.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.less.less <- unlist(lapply("accuracy.less.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.less.lessstd <- unlist(lapply("accuracy.less.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.less.lessstd2 <- unlist(lapply("accuracy.less.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.accuracy.svml1.none <- unlist(lapply("accuracy.svml1.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.svml1.std <- unlist(lapply("accuracy.svml1.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.svml1.less <- unlist(lapply("accuracy.svml1.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.svml1.lessstd <- unlist(lapply("accuracy.svml1.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.svml1.lessstd2 <- unlist(lapply("accuracy.svml1.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.accuracy.lrl1.none <- unlist(lapply("accuracy.lrl1.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lrl1.std <- unlist(lapply("accuracy.lrl1.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lrl1.less <- unlist(lapply("accuracy.lrl1.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lrl1.lessstd <- unlist(lapply("accuracy.lrl1.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lrl1.lessstd2 <- unlist(lapply("accuracy.lrl1.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))

sim.mean.accuracy.lasso.none <- unlist(lapply("accuracy.lasso.none", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lasso.std <- unlist(lapply("accuracy.lasso.std", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lasso.less <- unlist(lapply("accuracy.lasso.less", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lasso.lessstd <- unlist(lapply("accuracy.lasso.lessstd", function(k) {rowMeans(sapply(sim.list, "[[", k))}))
sim.mean.accuracy.lasso.lessstd2 <- unlist(lapply("accuracy.lasso.lessstd2", function(k) {rowMeans(sapply(sim.list, "[[", k))}))


sim.mean.df <- data.frame(Dimensions = rep(P.log[-1], 20),
                          mean.numbetas = c(sim.mean.numbetas.less.none[-1],
                                            sim.mean.numbetas.less.std[-1],
                                            sim.mean.numbetas.less.less[-1],
                                            sim.mean.numbetas.less.lessstd[-1],
                                            sim.mean.numbetas.less.lessstd2[-1],
                                            sim.mean.numbetas.svml1.none[-1],
                                            sim.mean.numbetas.svml1.std[-1],
                                            sim.mean.numbetas.svml1.less[-1],
                                            sim.mean.numbetas.svml1.lessstd[-1],
                                            sim.mean.numbetas.svml1.lessstd2[-1],
                                            sim.mean.numbetas.lrl1.none[-1],
                                            sim.mean.numbetas.lrl1.std[-1],
                                            sim.mean.numbetas.lrl1.less[-1],
                                            sim.mean.numbetas.lrl1.lessstd[-1],
                                            sim.mean.numbetas.lrl1.lessstd2[-1],
                                            sim.mean.numbetas.lasso.none[-1],
                                            sim.mean.numbetas.lasso.std[-1],
                                            sim.mean.numbetas.lasso.less[-1],
                                            sim.mean.numbetas.lasso.lessstd[-1],
                                            sim.mean.numbetas.lasso.lessstd2[-1]),
                          mean.auc = c(sim.mean.auc.less.none[-1],
                                       sim.mean.auc.less.std[-1],
                                       sim.mean.auc.less.less[-1],
                                       sim.mean.auc.less.lessstd[-1],
                                       sim.mean.auc.less.lessstd2[-1],
                                       sim.mean.auc.svml1.none[-1],
                                       sim.mean.auc.svml1.std[-1],
                                       sim.mean.auc.svml1.less[-1],
                                       sim.mean.auc.svml1.lessstd[-1],
                                       sim.mean.auc.svml1.lessstd2[-1],
                                       sim.mean.auc.lrl1.none[-1],
                                       sim.mean.auc.lrl1.std[-1],
                                       sim.mean.auc.lrl1.less[-1],
                                       sim.mean.auc.lrl1.lessstd[-1],
                                       sim.mean.auc.lrl1.lessstd2[-1],
                                       sim.mean.auc.lasso.none[-1],
                                       sim.mean.auc.lasso.std[-1],
                                       sim.mean.auc.lasso.less[-1],
                                       sim.mean.auc.lasso.lessstd[-1],
                                       sim.mean.auc.lasso.lessstd2[-1]),
                          mean.accuracy = c(sim.mean.accuracy.less.none[-1],
                                            sim.mean.accuracy.less.std[-1],
                                            sim.mean.accuracy.less.less[-1],
                                            sim.mean.accuracy.less.lessstd[-1],
                                            sim.mean.accuracy.less.lessstd2[-1],
                                            sim.mean.accuracy.svml1.none[-1],
                                            sim.mean.accuracy.svml1.std[-1],
                                            sim.mean.accuracy.svml1.less[-1],
                                            sim.mean.accuracy.svml1.lessstd[-1],
                                            sim.mean.accuracy.svml1.lessstd2[-1],
                                            sim.mean.accuracy.lrl1.none[-1],
                                            sim.mean.accuracy.lrl1.std[-1],
                                            sim.mean.accuracy.lrl1.less[-1],
                                            sim.mean.accuracy.lrl1.lessstd[-1],
                                            sim.mean.accuracy.lrl1.lessstd2[-1],
                                            sim.mean.accuracy.lasso.none[-1],
                                            sim.mean.accuracy.lasso.std[-1],
                                            sim.mean.accuracy.lasso.less[-1],
                                            sim.mean.accuracy.lasso.lessstd[-1],
                                            sim.mean.accuracy.lasso.lessstd2[-1]),
                          Method = c(rep("LESS", (length(P.log) - 1) * 5),
                                     rep("SVM", (length(P.log) - 1) * 5),
                                     rep("LogReg", (length(P.log) - 1) * 5),
                                     rep("LASSO", (length(P.log) - 1) * 5)),
                          Scaling = rep(c(rep("none", (length(P.log) - 1)),
                                          rep("std", (length(P.log) - 1)),
                                          rep("less", (length(P.log) - 1)),
                                          rep("lessstd", (length(P.log) - 1)),
                                          rep("lessstd2", (length(P.log) - 1))), 4))

save(sim.list, sim.mean.df, total.time, file = paste0("Simulation1_var=", S, ".RData"))


#### Plot results ####

# # Colors
groups <- 4
cols <- hcl(h = seq(15, 375, length = groups + 1), l = 65, c = 100)[1:groups]
# plot(1:groups, pch = 16, cex = 7, col = cols)
# plot(c(2:6, 9:13, 17:21, 24:28), pch = 16, cex = 7, col = cols[c(2:6, 9:13, 17:21, 24:28)])
# cols

# Relevel factors for correct order in plots
sim.mean.df$Method <- factor(sim.mean.df$Method, levels = c("LASSO", "LogReg", "SVM", "LESS"))
sim.mean.df$Scaling <- factor(sim.mean.df$Scaling, levels = c("none", "std", "less", "lessstd", "lessstd2"))

# Remove LESS_none and LESS_std
sim.mean.df <- subset(sim.mean.df, 
                      !((sim.mean.df$Method == "LESS" & sim.mean.df$Scaling == "none") |
                          (sim.mean.df$Method == "LESS" & sim.mean.df$Scaling == "std")))

# Plot mean sparseness (log scale)
ggsave(paste0("Simulation1_var=", S, "_Sparseness_loglines.png"),
       ggplot(sim.mean.df, aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

# Plot test AUC (log scale)
ggsave(paste0("Simulation1_var=", S, "_AUC_loglines.png"),
       ggplot(sim.mean.df, aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(),
       width = 150, height = 150, units = "mm")

# Plot test accuracy (log scale)
ggsave(paste0("Simulation1_var=", S, "_Accuracy_loglines.png"),
       ggplot(sim.mean.df, aes(x = Dimensions)) +
         geom_line(aes(y = mean.accuracy, colour = Method, linetype = Scaling), size = 1) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         # ggtitle(paste0("Test Accuracy (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Accuracy") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")


### Sparseness plots per classification method #################################
ggsave(paste0("sim1_method/Simulation1_var=", S, "_Sparseness_loglines_LESS.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "LESS", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[1]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_method/Simulation1_var=", S, "_Sparseness_loglines_SVM.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "SVM", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[4]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_method/Simulation1_var=", S, "_Sparseness_loglines_LogReg.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "LogReg", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[3]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_method/Simulation1_var=", S, "_Sparseness_loglines_LASSO.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "LASSO", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[2]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")



### AUC plots per classification method ########################################
ggsave(paste0("sim1_method/Simulation1_var=", S, "_AUC_loglines_LESS.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "LESS", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[1]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_method/Simulation1_var=", S, "_AUC_loglines_SVM.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "SVM", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[4]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_method/Simulation1_var=", S, "_AUC_loglines_LogReg.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "LogReg", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[3]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_method/Simulation1_var=", S, "_AUC_loglines_LASSO.png"),
       ggplot(sim.mean.df[sim.mean.df$Method == "LASSO", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[2]) +
         scale_linetype_manual(values = c("dotted", "solid", "dashed", "longdash", "twodash"),
                               labels = unname(TeX(c("$\\textit{x}$", "$\\textit{z}$", "$\\textit{\\mu_k}$", 
                                                     "$\\textit{\\mu_k \\sigma^2_k}$", 
                                                     "$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")



### Sparseness plots per scaling method ########################################
ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_Sparseness_loglines_none.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "none", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted"),
                               labels = unname(TeX(c("$\\textit{x}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_Sparseness_loglines_std.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "std", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("solid"),
                               labels = unname(TeX(c("$\\textit{z}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_Sparseness_loglines_les.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "less", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dashed"),
                               labels = unname(TeX(c("$\\textit{\\mu_k}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_Sparseness_loglines_lessstd.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "lessstd", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("longdash"),
                               labels = unname(TeX(c("$\\textit{\\mu_k \\sigma^2_k}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_Sparseness_loglines_lessstd2.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "lessstd2", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.numbetas, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("twodash"),
                               labels = unname(TeX(c("$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Model Sparseness (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("Number of selected variables") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")


### AUC plots per scaling method ###############################################
ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_AUC_loglines_none.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "none", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dotted"),
                               labels = unname(TeX(c("$\\textit{x}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_AUC_loglines_std.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "std", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("solid"),
                               labels = unname(TeX(c("$\\textit{z}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_AUC_loglines_les.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "less", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("dashed"),
                               labels = unname(TeX(c("$\\textit{\\mu_k}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_AUC_loglines_lessstd.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "lessstd", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("longdash"),
                               labels = unname(TeX(c("$\\textit{\\mu_k \\sigma^2_k}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

ggsave(paste0("sim1_scaling/Simulation1_var=", S, "_AUC_loglines_lessstd2.png"),
       ggplot(sim.mean.df[sim.mean.df$Scaling == "lessstd2", ], 
              aes(x = Dimensions)) +
         geom_line(aes(y = mean.auc, colour = Method, linetype = Scaling), size = 1) +
         scale_colour_manual(values = cols[c(2, 3, 4, 1)]) +
         scale_linetype_manual(values = c("twodash"),
                               labels = unname(TeX(c("$\\textit{\\mu_k \\bar{\\sigma}^2}$")))) +
         scale_x_log10() +
         annotation_logticks(sides = "b") +
         # ggtitle(paste0("Test AUC (mean over ", K, " simulations)")) +
         xlab("Number of variables") +
         ylab("AUC") +
         guides(color = guide_legend(order = 1, keywidth = 2.5),
                linetype = guide_legend(order = 2, keywidth = 2.5)) +
         theme_bw(), 
       width = 150, height = 150, units = "mm")

send_telegram_message(text = paste0("Simulation1_var=", S, " is finished!"),
                      chat_id = "441084295",
                      bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")

#### END ####