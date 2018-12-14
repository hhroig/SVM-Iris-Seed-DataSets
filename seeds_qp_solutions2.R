## Exercises on SVM (2nd HW: FDA)
## SVM Classification Solving the Quadratic Problem with "quadprog" and Linear K.
## Seeds Data Set (http://archive.ics.uci.edu/ml/datasets/seeds)
## by Harold A. Hernández Roig
## Inspired on code by R. Walker (https://gist.github.com/rwalk/64f1365f7a2470c498a4)


## About the dataset:
# Attributes:
# 1. area A, 
# 2. perimeter P, 
# 3. compactness C = 4*pi*A/P^2, 
# 4. length of kernel, 
# 5. width of kernel, 
# 6. asymmetry coefficient 
# 7. length of kernel groove. 
# Classe: Kama, Rosa and Canadian

## Packages
library(quadprog)
library(ggplot2)

## Load Data:
dataset = read.delim(file = "seeds_dataset.txt", 
                     header = FALSE, sep = "\t",
                     dec = ".")
dataset = na.omit(dataset)
colnames(dataset) <- c("A", "P", "C", "lengthK", "widthK", "asym", "groove", "class")
dataset$class = as.factor(dataset$class)
summary(dataset)

## Split Data: train 80% /test 20% to deal with overfitting: 120obs./30obs. for train/test resp.
set.seed(123)
split = sample.split(dataset$class, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

## Select the proper variables, C and WidthK:
training_set = training_set[,c(3,5,8)] 
test_set = test_set[,c(3,5,8)]

## Code for QP solver of SVM:
qp.svm.fit = function(n, X, y){
  # Dual method of Goldfarb and Idnani (1982, 1983) for solving quadratic programming
  # problems of the form min(-d^T b + 1/2 b^T D b) with the constraints A^T b >= b_0.
  
  # ... build the system matrices
  d <- matrix(1, nrow=n)
  b0 <- rbind( matrix(0, nrow=1, ncol=1) , matrix(0, nrow=n, ncol=1) )
  A <- t(rbind(matrix(y, nrow=1, ncol=n), diag(nrow=n)))
  
  # ...perturb matrix D in order for it to be symmetric positive definite:
  eps <- 5e-4
  Q <- sapply(1:n, function(i) y[i]*t(X)[,i])
  D <- t(Q)%*%Q
  
  # ... call the QP solver:
  sol <- solve.QP(D +eps*diag(n), d, A, b0, meq=1, factorized=FALSE)
  qpsol <- matrix(sol$solution, nrow=n) # alphas!
  return(qpsol)
}

## Code to extract and Plot the decision boundary:
# ...build the support vectors, slopes, and intercepts
findLine <- function(a, y, X){
  nonzero <-  abs(a) > 1e-5
  W <- rowSums(sapply(which(nonzero), function(i) a[i]*y[i]*X[i,]))
  b <- mean(sapply(which(nonzero), function(i) X[i,]%*%W- y[i]))
  slope <- -W[1]/W[2]
  intercept <- b/W[2]
  return(c(intercept,slope))
}

## -------------------------------------------------------
##  Kama (1) vs. All
## Transform Data as needed for QP code:

X = as.matrix(training_set[,1:2]) 
n <- dim(X)[1]

y = array(-1, length(training_set[,3]))
y[training_set[,3] == "1"] = 1
y = as.matrix(y)

qpsol = qp.svm.fit(n, X, y)

qpline = findLine( qpsol, y, X)

# plot the results
plt <- ggplot(training_set, aes(x=C, y=widthK)) + 
  ggtitle("QP Solution: Kama (1) vs. All (linear Kernel)") +
  geom_point(aes(fill=factor(y)), size=3, pch=21) +
  geom_abline(intercept=qpline[1], slope=qpline[2], size=1, aes(color="quadprog"), show.legend) +
  scale_fill_manual(values=c("red","blue"), guide='none')
print(plt)

## -------------------------------------------------------
##  Rosa (2) vs. All
## Transform Data as needed for QP code:

X = as.matrix(training_set[,1:2]) 
n <- dim(X)[1]

y = array(-1, length(training_set[,3]))
y[training_set[,3] == "2"] = 1
y = as.matrix(y)

qpsol = qp.svm.fit(n, X, y)

qpline = findLine( qpsol, y, X)

# plot the results
plt <- ggplot(training_set, aes(x=C, y=widthK)) + 
  ggtitle("QP Solution: Rosa (2) vs. All (linear Kernel)") +
  geom_point(aes(fill=factor(y)), size=3, pch=21) +
  geom_abline(intercept=qpline[1], slope=qpline[2], size=1, aes(color="quadprog"), show.legend) +
  scale_fill_manual(values=c("red","blue"), guide='none')
print(plt)

## -------------------------------------------------------
##  Canadian (3) vs. All
## Transform Data as needed for QP code:

X = as.matrix(training_set[,1:2]) 
n <- dim(X)[1]

y = array(-1, length(training_set[,3]))
y[training_set[,3] == "3"] = 1
y = as.matrix(y)

qpsol = qp.svm.fit(n, X, y)

qpline = findLine( qpsol, y, X)

# plot the results
plt <- ggplot(training_set, aes(x=C, y=widthK)) + 
  ggtitle("QP Solution: Canadian (3) vs. All (linear Kernel)") +
  geom_point(aes(fill=factor(y)), size=3, pch=21) +
  geom_abline(intercept=qpline[1], slope=qpline[2], size=1, aes(color="quadprog"), show.legend) +
  scale_fill_manual(values=c("red","blue"), guide='none')
print(plt)