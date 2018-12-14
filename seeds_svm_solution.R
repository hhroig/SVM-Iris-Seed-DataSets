## Exercises on SVM (2nd HW: FDA)
## SVM Classification using package "e1071" 
## Seeds Data Set (http://archive.ics.uci.edu/ml/datasets/seeds)
## by Harold A. Hernández Roig

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

# Packages
if (!require(caTools))
  install.packages("caTools")
library(caTools) # to split data into train/test

if (!require(ggplot2))
  install.packages("ggplot2")
library(ggplot2) # nice plots...

if (!require(GGally))
  install.packages("GGally")
library(GGally)  # also plotting tools

if (!require(e1071))
  install.packages("e1071")
library(e1071)   # classical package for svm


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

# Some exploratory visualization 
ggpairs(training_set, ggplot2::aes(colour = class, alpha = 0.4))

## Fixing just 2 covariates for simplicity, nonlinearity and graphical representations...
training_set = training_set[,c(3,5,8)]
test_set = test_set[,c(3,5,8)]


# ----------------------------------------------------------------------
##  LINEAR KERNEL
model.l <- svm(class~., data = training_set,
               type="C-classification",kernel="linear",probability = TRUE)
print(model.l)
summary(model.l)

# Compute decision values and probabilities:
pred.l <- predict(model.l, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.l, "probabilities")
attr(pred.l, "decision.values")

# Check accuracy:
table(pred.l, test_set$class)
mean(pred.l == test_set$class) # Rate of well classified! (predicted)

## Tuning cost:
tune.l <- tune.svm(class~., data = training_set,
                   type="C-classification",kernel="linear",probability = TRUE, cost = seq(0.1,10,by=0.1))
summary(tune.l)
optimal.cost.l = tune.l$best.parameters$cost

## Tuned Model:

model.l.tuned <- svm(class~., data = training_set,
                     type="C-classification",kernel="linear",probability = TRUE, cost = optimal.cost.l) 
print(model.l.tuned)
summary(model.l.tuned)

# Compute decision values and probabilities:
pred.l.tuned <- predict(model.l.tuned, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.l.tuned, "probabilities")
attr(pred.l.tuned, "decision.values")

# Check accuracy:
table(pred.l.tuned, test_set$class)
mean(pred.l.tuned == test_set$class) # Rate of well classified! (predicted)

plot(model.l.tuned, test_set)
plot(model.l.tuned, training_set)

# ----------------------------------------------------------------------
##  POLYNOMIAL KERNEL
model.p <- svm(class ~ ., data = training_set,
               type="C-classification",kernel="polynomial",probability = TRUE)

print(model.p)
summary(model.p)

# Compute decision values and probabilities:
pred.p <- predict(model.p, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.p, "probabilities")
attr(pred.p, "decision.values")

# Check accuracy:
table(pred.p, test_set$class)
mean(pred.p == test_set$class) # Rate of well classified!

# Plot Decision Boundaries:
plot(model.p, test_set)
plot(model.p, training_set)

# Tuning cost, degree, gamma and coef0:
tune.p <- tune.svm(class ~ ., data = training_set,
                   type="C-classification",kernel="polynomial",probability = TRUE,
                   #cost = seq(0.1,10,by=0.1),degree = seq(2,10,by=1))
                   cost = seq(1,10,by=1), coef0 = c(0,1), degree = seq(3,4,by=1), gamma = seq(0.6,1,by=0.2))
summary(tune.p)
optimal.cost.p = tune.p$best.parameters$cost
optimal.coef0.p = tune.p$best.parameters$coef0
optimal.degree.p = tune.p$best.parameters$degree
optimal.gamma.p = tune.p$best.parameters$gamma

## Tuned Model:

model.p.tuned <- svm(class ~ ., data = training_set,
                     type="C-classification",kernel="polynomial",probability = TRUE,
                     cost = optimal.cost.p, coef0 = optimal.coef0.p,
                     degree = optimal.degree.p, gamma = optimal.gamma.p)

print(model.p.tuned)
summary(model.p.tuned)

# Compute decision values and probabilities:
pred.p.tuned <- predict(model.p.tuned, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.p.tuned, "probabilities")
attr(pred.p.tuned, "decision.values")

# Check accuracy:
table(pred.p.tuned, test_set$class)
mean(pred.p.tuned == test_set$class) # Rate of well classified!

# Plot Decision Boundaries:
plot(model.p.tuned, test_set)
plot(model.p.tuned, training_set)

# ----------------------------------------------------------------------
##  RBF (GAUSSIAN) KERNEL

model.g <- svm(class~., data = training_set,
               type="C-classification",kernel="radial",probability = TRUE)

print(model.g)
summary(model.g)

# Compute decision values and probabilities:
pred.g <- predict(model.g, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.g, "probabilities")
attr(pred.g, "decision.values")

# Check accuracy:
table(pred.g, test_set$class)
mean(pred.g == test_set$class) # Rate of well classified!

# Plot Decision Boundaries:
plot(model.g, test_set)
plot(model.g, training_set)

# Tuning cost, degree, gamma and coef0:
tune.g <- tune.svm(class ~ ., data = training_set,
                   type="C-classification",kernel="radial",probability = TRUE,
                   cost = seq(1,10,by=1), gamma = seq(0,10,by=.5))
summary(tune.g)
# plot(tune.g)
optimal.cost.g = tune.g$best.parameters$cost
optimal.gamma.g = tune.g$best.parameters$gamma

## Tuned Model:
model.g.tuned <- svm(class ~ ., data = training_set,
                     type="C-classification",kernel="radial",probability = TRUE,
                     cost = optimal.cost.g, gamma = optimal.gamma.g)

print(model.g.tuned)
summary(model.g.tuned)

# Compute decision values and probabilities:
pred.g.tuned <- predict(model.g.tuned, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.g.tuned, "probabilities")
attr(pred.g.tuned, "decision.values")

# Check accuracy:
table(pred.g.tuned, test_set$class)
mean(pred.g.tuned == test_set$class) # Rate of well classified!

# Plot Decision Boundaries:
plot(model.g.tuned, test_set)
plot(model.g.tuned, training_set)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
##  MIXTURE OF KERNELS
library(kernlab)
# Average Kernel = RBF Kernel + Polynomial Kernel
kfunction2 <- function(sigma, degree, scale, offset)
{
  k <- function (x,y)
  {
    1/2*exp(-sigma*sum((x-y)^2)) + 1/2*(scale*sum((x)*(y))+offset)^degree
  }
  class(k) <- "kernel"
  k
}

svp <- ksvm(class ~ ., data = training_set,
            type="C-svc", kernel=kfunction2(4, 4, 1, 1), C = 1, prob.model=TRUE)
svp

## get fitted values
fit.svp = fitted(svp)
## Test on the training set with probabilities as output
pred.svp = predict(svp, test_set, type = "response")

# Check accuracy:
# ... on test
table(pred.svp, test_set$class)
mean(pred.svp == test_set$class) # Rate of well classified!
# ... on train
table(fit.svp, training_set$class)
mean(fit.svp == training_set$class) # Rate of well classified!
