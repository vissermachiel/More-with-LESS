# LESSTWOC: Two Class LESS-Classifier
#
# W = LESSTWOC(A,C)
#
# INPUT
#   A       Dataset
#   C       Regularization parameter, C >= 0
#           default: C = 1
# OUTPUT
#   W       LESS classifier

# Copyright: Cor J. Veenman

require(lpSolve)

mapmeans <- function(DF, M){
  # DF - the dataframe 
  # M  - 2 x p matrix with the means per class
  n <- nrow(DF)
  p <- ncol(DF)
  m1 <- matrix(rep(M[1,], times = n), nrow = n, byrow = TRUE)
  m2 <- matrix(rep(M[2,], times = n), nrow = n, byrow = TRUE)
  
  return((DF-m1)^2 - (DF-m2)^2)
}

mapmeansstd <- function(DF, M, S){
  # DF - the dataframe 
  # M  - 2 x p matrix with the means per class
  # S  - 2 x p matrix with the variance per class
  n <- nrow(DF)
  p <- ncol(DF)
  m1 <- matrix(rep(M[1,], times = n), nrow = n, byrow = TRUE)
  m2 <- matrix(rep(M[2,], times = n), nrow = n, byrow = TRUE)
  s1 <- matrix(rep(S[1,], times = n), nrow = n, byrow = TRUE)
  s2 <- matrix(rep(S[2,], times = n), nrow = n, byrow = TRUE)
  
  return((DF-m1)^2 / s1 - (DF-m2)^2 / s2)
}

mapmeansstd2 <- function(DF, M, S2){
  # DF - the dataframe 
  # M  - 2 x p matrix with the means per class
  # S  - 1 x p matrix with the variance after subtracting the class means
  n <- nrow(DF)
  p <- ncol(DF)
  m1 <- matrix(rep(M[1,], times = n), nrow = n, byrow = TRUE)
  m2 <- matrix(rep(M[2,], times = n), nrow = n, byrow = TRUE)
  s  <- matrix(rep(S2, times = n), nrow = n, byrow = TRUE)

  return((DF-m1)^2 / s - (DF-m2)^2 / s)
}

mapstd <- function(DF, M, SD){
  # DF - the dataframe 
  # M  - 1 x p matrix with the means for each variable
  # S  - 1 x p matrix with the variance for each variable
  n <- nrow(DF)
  p <- ncol(DF)
  m <- matrix(rep(M, times = n), nrow = n, byrow = TRUE)
  sd <- matrix(rep(SD, times = n), nrow = n, byrow = TRUE)
  
  return((DF - m) / sd ^ as.logical(sd))
}



lesstwoc <- function(DF, C = 1){
  # DF    - a data frame. The last column must be a factor representating 
  #         the binary target, the other columns must be numeric predictors.
  # C     - a number or a vector of length 2 (one number for each class) of
  #         the target. C represents the misallocation that is allowed.
  # Purpose: train the LESS classifier. 
  
  # collecting data about the target
  labs <- DF[,ncol(DF)]
  clasval <- levels(labs) # first level = +1, second level = -1
  indclas1 <- which(labs == clasval[1])
  indclas2 <- setdiff(1:nrow(DF), indclas1)
  classObjects <- c(length(indclas1), length(indclas2))
  
  # general info about input data
  numObjects <- nrow(DF)
  numFeatures <- ncol(DF) - 1
  
  # estimate class means
  M <- matrix(0.0, nrow = 2, ncol = numFeatures)
  
  M[1,] <- apply(as.matrix(DF[indclas1,-ncol(DF)]),2,mean)
  M[2,] <- apply(as.matrix(DF[indclas2,-ncol(DF)]),2,mean)
  
  # setting C properly
  if(is.na(C)) {
    C = 1
  }
  if(length(C) == 1){
    C = numObjects / 2 * c(C/classObjects[1], C/classObjects[2])
  } else{
    if(length(C) != 2) stop('C should be a scalar or a vector with length
                            equal to the number of classes')
  }
    
  ## Set the optimization function and C parameter
  fvector <- rep(1, times = numFeatures)
  Cvector <- c(rep(C[1], times = classObjects[1]),rep(C[2], times = classObjects[2]))  
  
  f <- c(fvector, Cvector)
  
  # setup constraints
  b <- rep(1, times = numObjects)
  IX <- c(indclas1, indclas2)

  d  <- mapmeans(DF[IX, -ncol(DF), drop = FALSE], M)
  
  y <- c(rep(-1, times = classObjects[1]), rep(1, times = classObjects[2])) 
  
  A <- matrix(rep(y, numFeatures), ncol = numFeatures, byrow = FALSE) * d
  
  A <- cbind(A, diag(rep(1, times = numObjects)))
  
  f.dir <- rep(">=", times = numObjects)
  
  
  # --- LINPROG CALL ---
  reslp <- lp(objective.in = f, 
              const.mat = A, 
              const.dir = f.dir, 
              const.rhs = b)
  
  outlp <- list(reslp$solution[1:numFeatures],
              reslp$objval,
              reslp$status)
  
  out <- list(status = "trained",
              model = list(beta = outlp[[1]],
                           M = M),
              target_levels = clasval,
              num_features = numFeatures,
              numClasses = 2
              )
  return(out)
}

predict.less <- function(MODEL, NEWDATA){
  # MODEL   - a model as is the output of the function lesstwoc
  # NEWDATA - a data frame with columns the predictors for the model
  #
  NEWDATA <- as.matrix(NEWDATA)
  y <- data.frame(score = rep(0.0, times = nrow(NEWDATA)),
                  prediction = rep(" ", times = nrow(NEWDATA)))
  levels <- MODEL$target_levels
  
  beta <- MODEL$model[[1]]
  M    <- MODEL$model[[2]]
  
  z <- mapmeans(NEWDATA, M) %*% beta
  
  y[,1] <- z
  y[,2] <- factor(ifelse(z<0, levels[1], levels[2]))
  
  return(y)
}

################################################################################

lesstwoc_lessstd <- function(DF, C = 1){
  # DF    - a data frame. The last column must be a factor representating 
  #         the binary target, the other columns must be numeric predictors.
  # C     - a number or a vector of length 2 (one number for each class) of
  #         the target. C represents the misallocation that is allowed.
  # Purpose: train the LESS classifier. 
  
  # collecting data about the target
  labs <- DF[, ncol(DF)]
  clasval <- levels(labs) # first level = +1, second level = -1
  indclas1 <- which(labs == clasval[1])
  indclas2 <- setdiff(1:nrow(DF), indclas1)
  classObjects <- c(length(indclas1), length(indclas2))
  
  # general info about input data
  numObjects <- nrow(DF)
  numFeatures <- ncol(DF) - 1
  
  # estimate class means
  M <- matrix(0.0, nrow = 2, ncol = numFeatures)
  M[1,] <- apply(as.matrix(DF[indclas1,-ncol(DF)]), 2, mean)
  M[2,] <- apply(as.matrix(DF[indclas2,-ncol(DF)]), 2, mean)
  
  # estimate class variances
  S <- matrix(0.0, nrow = 2, ncol = numFeatures)
  S[1, ] <- apply(as.matrix(DF[indclas1,-ncol(DF)]), 2, var)
  S[2, ] <- apply(as.matrix(DF[indclas2,-ncol(DF)]), 2, var)
  S <- ifelse(S == 0, unique(sort(S))[2], S) # When var = 0
  
  # setting C properly
  if(is.na(C)) {
    C = 1
  }
  if(length(C) == 1){
    C = numObjects / 2 * c(C/classObjects[1], C/classObjects[2])
  } else{
    if(length(C) != 2) stop('C should be a scalar or a vector with length
                            equal to the number of classes')
  }
  
  ## Set the optimization function and C parameter
  fvector <- rep(1, times = numFeatures)
  Cvector <- c(rep(C[1], times = classObjects[1]),rep(C[2], times = classObjects[2]))  
  
  f <- c(fvector, Cvector)
  
  # setup constraints
  b <- rep(1, times = numObjects)
  IX <- c(indclas1, indclas2)
  
  d  <- mapmeansstd(DF[IX, -ncol(DF), drop = FALSE], M, S)
  
  y <- c(rep(-1, times = classObjects[1]), rep(1, times = classObjects[2])) 
  
  A <- matrix(rep(y, numFeatures), ncol = numFeatures, byrow = FALSE) * d
  
  A <- cbind(A, diag(rep(1, times = numObjects)))
  
  f.dir <- rep(">=", times = numObjects)
  
  
  # --- LINPROG CALL ---
  reslp <- lp(objective.in = f, 
              const.mat = A, 
              const.dir = f.dir, 
              const.rhs = b)
  
  outlp <- list(reslp$solution[1:numFeatures],
                reslp$objval,
                reslp$status)
  
  out <- list(status = "trained",
              model = list(beta = outlp[[1]],
                           M = M,
                           S = S),
              target_levels = clasval,
              num_features = numFeatures,
              numClasses = 2
  )
  return(out)
}



lesstwoc_lessstd2 <- function(DF, C = 1){
  # DF    - a data frame. The last column must be a factor representating 
  #         the binary target, the other columns must be numeric predictors.
  # C     - a number or a vector of length 2 (one number for each class) of
  #         the target. C represents the misallocation that is allowed.
  # Purpose: train the LESS classifier. 
  
  # collecting data about the target
  labs <- DF[,ncol(DF)]
  clasval <- levels(labs) # first level = +1, second level = -1
  indclas1 <- which(labs == clasval[1])
  indclas2 <- setdiff(1:nrow(DF), indclas1)
  classObjects <- c(length(indclas1), length(indclas2))
  
  # general info about input data
  numObjects <- nrow(DF)
  numFeatures <- ncol(DF) - 1
  
  # estimate class means
  M <- matrix(0.0, nrow = 2, ncol = numFeatures)
  M[1, ] <- apply(as.matrix(DF[indclas1, -ncol(DF)]), 2, mean)
  M[2, ] <- apply(as.matrix(DF[indclas2, -ncol(DF)]), 2, mean)
  
  # estimate variance after subtracting class means
  S2 <- matrix(apply(rbind(DF[indclas1, -ncol(DF)] -
                             matrix(rep(M[1, ], 
                                        times = length(indclas1)),
                                    nrow = length(indclas1),
                                    byrow = TRUE),
                           DF[indclas2, -ncol(DF)] -
                             matrix(rep(M[2, ],
                                        times = length(indclas2)),
                                    nrow = length(indclas2),
                                    byrow = TRUE)),
                     2, var),
               nrow = 1, ncol = numFeatures)
  S2 <- ifelse(S2 == 0, unique(sort(S2))[2], S2) # When var = 0
  
  # setting C properly
  if(is.na(C)) {
    C = 1
  }
  if(length(C) == 1){
    C = numObjects / 2 * c(C/classObjects[1], C/classObjects[2])
  } else{
    if(length(C) != 2) stop('C should be a scalar or a vector with length
                            equal to the number of classes')
  }
  
  ## Set the optimization function and C parameter
  fvector <- rep(1, times = numFeatures)
  Cvector <- c(rep(C[1], times = classObjects[1]),rep(C[2], times = classObjects[2]))  
  
  f <- c(fvector, Cvector)
  
  # setup constraints
  b <- rep(1, times = numObjects)
  IX <- c(indclas1, indclas2)
  
  d  <- mapmeansstd2(DF[IX, -ncol(DF), drop = FALSE], M, S2)
  
  y <- c(rep(-1, times = classObjects[1]), rep(1, times = classObjects[2])) 
  
  A <- matrix(rep(y, numFeatures), ncol = numFeatures, byrow = FALSE) * d
  
  A <- cbind(A, diag(rep(1, times = numObjects)))
  
  f.dir <- rep(">=", times = numObjects)
  
  
  # --- LINPROG CALL ---
  reslp <- lp(objective.in = f, 
              const.mat = A, 
              const.dir = f.dir, 
              const.rhs = b)
  
  outlp <- list(reslp$solution[1:numFeatures],
                reslp$objval,
                reslp$status)
  
  out <- list(status = "trained",
              model = list(beta = outlp[[1]],
                           M = M,
                           S2 = S2),
              target_levels = clasval,
              num_features = numFeatures,
              numClasses = 2
  )
  return(out)
}



lesstwoc_none <- function(DF, C = 1){
  # DF    - a data frame. The last column must be a factor representating 
  #         the binary target, the other columns must be numeric predictors.
  # C     - a number or a vector of length 2 (one number for each class) of
  #         the target. C represents the misallocation that is allowed.
  # Purpose: train the LESS classifier. 
  
  # collecting data about the target
  labs <- DF[,ncol(DF)]
  clasval <- levels(labs) # first level = +1, second level = -1
  indclas1 <- which(labs == clasval[1])
  indclas2 <- setdiff(1:nrow(DF), indclas1)
  classObjects <- c(length(indclas1), length(indclas2))
  
  # general info about input data
  numObjects <- nrow(DF)
  numFeatures <- ncol(DF) - 1
  
  # setting C properly
  if(is.na(C)) {
    C = 1
  }
  if(length(C) == 1){
    C = numObjects / 2 * c(C/classObjects[1], C/classObjects[2])
  } else{
    if(length(C) != 2) stop('C should be a scalar or a vector with length
                            equal to the number of classes')
  }
  
  ## Set the optimization function and C parameter
  fvector <- rep(1, times = numFeatures)
  Cvector <- c(rep(C[1], times = classObjects[1]),rep(C[2], times = classObjects[2]))  
  
  f <- c(fvector, Cvector)
  
  # setup constraints
  b <- rep(1, times = numObjects)
  IX <- c(indclas1, indclas2)
  
  d  <- DF[IX, -ncol(DF), drop = FALSE]
  
  y <- c(rep(-1, times = classObjects[1]), rep(1, times = classObjects[2])) 
  
  A <- matrix(rep(y, numFeatures), ncol = numFeatures, byrow = FALSE) * d
  
  A <- cbind(A, diag(rep(1, times = numObjects)))
  
  f.dir <- rep(">=", times = numObjects)
  
  
  # --- LINPROG CALL ---
  reslp <- lp(objective.in = f, 
              const.mat = A, 
              const.dir = f.dir, 
              const.rhs = b)
  
  outlp <- list(reslp$solution[1:numFeatures],
                reslp$objval,
                reslp$status)
  
  out <- list(status = "trained",
              model = list(beta = outlp[[1]]),
              target_levels = clasval,
              num_features = numFeatures,
              numClasses = 2
  )
  return(out)
}


lesstwoc_std <- function(DF, C = 1){
  # DF    - a data frame. The last column must be a factor representating 
  #         the binary target, the other columns must be numeric predictors.
  # C     - a number or a vector of length 2 (one number for each class) of
  #         the target. C represents the misallocation that is allowed.
  # Purpose: train the LESS classifier. 
  
  # collecting data about the target
  labs <- DF[, ncol(DF)]
  clasval <- levels(labs) # first level = +1, second level = -1
  indclas1 <- which(labs == clasval[1])
  indclas2 <- setdiff(1:nrow(DF), indclas1)
  classObjects <- c(length(indclas1), length(indclas2))
  
  # general info about input data
  numObjects <- nrow(DF)
  numFeatures <- ncol(DF) - 1
  
  # estimate variable means
  M <- matrix(0.0, nrow = 1, ncol = numFeatures)
  M[1, ] <- colMeans(DF[, -ncol(DF)])
  
  # estimate standard deviation for each variable
  SD <- matrix(0.0, nrow = 1, ncol = numFeatures)
  SD[1, ] <- apply(as.matrix(DF[, -ncol(DF)]), 2, sd)
  SD <- ifelse(SD == 0, unique(sort(SD))[2], SD) # When sd = 0
  
  # setting C properly
  if(is.na(C)) {
    C = 1
  }
  if(length(C) == 1){
    C = numObjects / 2 * c(C/classObjects[1], C/classObjects[2])
  } else{
    if(length(C) != 2) stop('C should be a scalar or a vector with length
                            equal to the number of classes')
  }
  
  ## Set the optimization function and C parameter
  fvector <- rep(1, times = numFeatures)
  Cvector <- c(rep(C[1], times = classObjects[1]), rep(C[2], times = classObjects[2]))  
  
  f <- c(fvector, Cvector)
  
  # setup constraints
  b <- rep(1, times = numObjects)
  IX <- c(indclas1, indclas2)
  
  d  <- mapstd(DF[IX, -ncol(DF), drop = FALSE], M, SD)
  
  y <- c(rep(-1, times = classObjects[1]), rep(1, times = classObjects[2])) 
  
  A <- matrix(rep(y, numFeatures), ncol = numFeatures, byrow = FALSE) * d
  
  A <- cbind(A, diag(rep(1, times = numObjects)))
  
  f.dir <- rep(">=", times = numObjects)
  
  # --- LINPROG CALL ---
  reslp <- lp(objective.in = f, 
              const.mat = A, 
              const.dir = f.dir, 
              const.rhs = b)
  
  outlp <- list(reslp$solution[1:numFeatures],
                reslp$objval,
                reslp$status)
  
  out <- list(status = "trained",
              model = list(beta = outlp[[1]],
                           M = M,
                           SD = SD),
              target_levels = clasval,
              num_features = numFeatures,
              numClasses = 2
  )
  return(out)
}


predict.less_lessstd <- function(MODEL, NEWDATA){
  # MODEL   - a model as is the output of the function lesstwoc
  # NEWDATA - a data frame with columns the predictors for the model
  #
  NEWDATA <- as.matrix(NEWDATA)
  y <- data.frame(score = rep(0.0, times = nrow(NEWDATA)),
                  prediction = rep(" ", times = nrow(NEWDATA)))
  levels <- MODEL$target_levels
  
  beta <- MODEL$model[[1]]
  M    <- MODEL$model[[2]]
  S    <- MODEL$model[[3]]
  
  z <- mapmeansstd(NEWDATA, M, S) %*% beta
  
  y[,1] <- z
  y[,2] <- factor(ifelse(z<0, levels[1], levels[2]))
  
  return(y)
}


predict.less_lessstd2 <- function(MODEL, NEWDATA){
  # MODEL   - a model as is the output of the function lesstwoc
  # NEWDATA - a data frame with columns the predictors for the model
  #
  NEWDATA <- as.matrix(NEWDATA)
  y <- data.frame(score = rep(0.0, times = nrow(NEWDATA)),
                  prediction = rep(" ", times = nrow(NEWDATA)))
  levels <- MODEL$target_levels
  
  beta <- MODEL$model[[1]]
  M    <- MODEL$model[[2]]
  S2   <- MODEL$model[[3]]
  
  z <- mapmeansstd2(NEWDATA, M, S2) %*% beta
  
  y[,1] <- z
  y[,2] <- factor(ifelse(z<0, levels[1], levels[2]))
  
  return(y)
}


predict.less_none <- function(MODEL, NEWDATA){
  # MODEL   - a model as is the output of the function lesstwoc
  # NEWDATA - a data frame with columns the predictors for the model
  #
  NEWDATA <- as.matrix(NEWDATA)
  y <- data.frame(score = rep(0.0, times = nrow(NEWDATA)),
                  prediction = rep(" ", times = nrow(NEWDATA)))
  levels <- MODEL$target_levels
  
  beta <- MODEL$model[[1]]
  
  z <- NEWDATA %*% beta
  
  y[,1] <- z
  y[,2] <- factor(ifelse(z<0, levels[1], levels[2]))
  
  return(y)
}


predict.less_std <- function(MODEL, NEWDATA){
  # MODEL   - a model as is the output of the function lesstwoc
  # NEWDATA - a data frame with columns the predictors for the model
  #
  NEWDATA <- as.matrix(NEWDATA)
  y <- data.frame(score = rep(0.0, times = nrow(NEWDATA)),
                  prediction = rep(" ", times = nrow(NEWDATA)))
  levels <- MODEL$target_levels
  
  beta <- MODEL$model[[1]]
  M    <- MODEL$model[[2]]
  SD   <- MODEL$model[[3]]
  
  z <- mapstd(NEWDATA, M, SD) %*% beta

  y[, 1] <- z
  y[, 2] <- factor(ifelse(z<0, levels[1], levels[2]))
  
  return(y)
}
