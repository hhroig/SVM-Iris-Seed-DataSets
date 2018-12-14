## Exercises on SVM (2nd HW: FDA)
## SVM Classification using package "e1071"
## Iris Data Set
## by Harold A. Hernández Roig

## First steps are based on: https://rpubs.com/piyushpallav/irisSVMclassifier

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


## Decision Plot from: http://michael.hahsler.net/SMU/EMIS7332/R/viz_classifier.html
decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  
  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}

## Load Data:
dataset = iris
summary(dataset)

## Split Data: train 80% /test 20% to deal with overfitting: 120obs./30obs. for train/test resp.
set.seed(123)
split = sample.split(dataset$Species, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Some exploratory visualization 
 ggpairs(dataset, ggplot2::aes(colour = Species, alpha = 0.4))

# Some basic conclusions: Histograms for Petal Width and Length show a clear separation of Setosa,
# while there is some overlapping (not too much) between the other 2 species... It seems like a good
# approach only to take into account these two variables to classify!

# ----------------------------------------------------------------------
##  LINEAR KERNEL
model.l <- svm(Species ~ Petal.Width + Petal.Length, data = training_set,
             type="C-classification",kernel="linear",probability = TRUE)
print(model.l)
summary(model.l)

# Compute decision values and probabilities:
pred.l <- predict(model.l, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.l, "probabilities")
attr(pred.l, "decision.values")

# Check accuracy:
table(pred.l, test_set[,5])
mean(pred.l == test_set[,5]) # Rate of well classified! (predicted)

# Plot Decision Boundaries:
decisionplot(model.l, test_set[,3:5], class = "Species", main = "SVD (linear)")
plot(model.l, test_set[,3:5])
plot(model.l, training_set[,3:5])

## Tuning cost:
tune.l <- tune.svm(Species ~ Petal.Width + Petal.Length, data = training_set,
                     type="C-classification",kernel="linear",probability = TRUE, cost = seq(0.1,10,by=0.1))
summary(tune.l)
plot(tune.l)
optimal.cost.l = tune.l$best.parameters$cost

## Tuned Model:

model.l.tuned <- svm(Species ~ Petal.Width + Petal.Length, data = training_set,
               type="C-classification",kernel="linear",probability = TRUE, cost = optimal.cost.l) 
print(model.l.tuned)
summary(model.l.tuned)

# Compute decision values and probabilities:
pred.l.tuned <- predict(model.l.tuned, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.l.tuned, "probabilities")
attr(pred.l.tuned, "decision.values")

# Check accuracy:
table(pred.l.tuned, test_set[,5])
mean(pred.l.tuned == test_set[,5]) # Rate of well classified! (predicted)

# Plot Decision Boundaries:
decisionplot(model.l.tuned, test_set[,3:5], class = "Species", main = "SVD (linear)")
plot(model.l.tuned, test_set[,3:5])
plot(model.l.tuned, training_set[,3:5])


# ----------------------------------------------------------------------
##  POLYNOMIAL KERNEL
model.p <- svm(Species ~ Petal.Width + Petal.Length, data = training_set,
                     type="C-classification",kernel="polynomial",probability = TRUE)

print(model.p)
summary(model.p)

# Compute decision values and probabilities:
pred.p <- predict(model.p, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.p, "probabilities")
attr(pred.p, "decision.values")

# Check accuracy:
table(pred.p, test_set[,5])
mean(pred.p == test_set[,5]) # Rate of well classified!

# Plot Decision Boundaries:
decisionplot(model.p, test_set[,3:5], class = "Species", main = "SVD (polynomial)")
plot(model.p, test_set[,3:5])
plot(model.p, training_set[,3:5])

# Tuning cost, degree, gamma and coef0:
tune.p <- tune.svm(Species ~ Petal.Width + Petal.Length, data = training_set,
                   type="C-classification",kernel="polynomial",probability = TRUE,
                   #cost = seq(0.1,10,by=0.1),degree = seq(2,10,by=1))
                 cost = seq(0.5,5,by=.5), coef0 = 0, degree = seq(3,9,by=1), gamma = seq(0.1,0.8,by=0.1))
summary(tune.p)

optimal.cost.p = tune.p$best.parameters$cost
optimal.coef0.p = tune.p$best.parameters$coef0
optimal.degree.p = tune.p$best.parameters$degree
optimal.gamma.p = tune.p$best.parameters$gamma

## Tuned Model:

model.p.tuned <- svm(Species ~ Petal.Width + Petal.Length, data = training_set,
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
table(pred.p.tuned, test_set[,5])
mean(pred.p.tuned == test_set[,5]) # Rate of well classified!

# Plot Decision Boundaries:
decisionplot(model.p.tuned, test_set[,3:5], class = "Species", main = "SVD (polynomial)")
plot(model.p.tuned, test_set[,3:5])
plot(model.p.tuned, training_set[,3:5])

# ----------------------------------------------------------------------
##  RBF (GAUSSIAN) KERNEL

model.g <- svm(Species ~ Petal.Width + Petal.Length, data = training_set,
             type="C-classification",kernel="radial",probability = TRUE)

print(model.g)
summary(model.g)

# Compute decision values and probabilities:
pred.g <- predict(model.g, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.g, "probabilities")
attr(pred.g, "decision.values")

# Check accuracy:
table(pred.g, test_set[,5])
mean(pred.g == test_set[,5]) # Rate of well classified!

# Plot Decision Boundaries:
decisionplot(model.g, test_set[,3:5], class = "Species", main = "SVD (RBF)")
plot(model.g, test_set[,3:5])
plot(model.g, training_set[,3:5])

# Tuning cost, degree, gamma and coef0:
tune.g <- tune.svm(Species ~ Petal.Width + Petal.Length, data = training_set,
                   type="C-classification",kernel="radial",probability = TRUE,
                   cost = seq(0.1,10,by=0.1), gamma = seq(0,10,by=.1))
summary(tune.g)
optimal.cost.g = tune.g$best.parameters$cost
optimal.gamma.g = tune.g$best.parameters$gamma

## Tuned Model:
model.g.tuned <- svm(Species ~ Petal.Width + Petal.Length, data = training_set,
               type="C-classification",kernel="radial",probability = TRUE,
               cost = optimal.cost.g, gamma = optimal.gamma.g)

print(model.g.tuned)
summary(model.g.tuned)

# Compute decision values and probabilities:
pred.g.tuned <- predict(model.g.tuned, test_set, decision.values = TRUE, probability = TRUE)
attr(pred.g.tuned, "probabilities")
attr(pred.g.tuned, "decision.values")

# Check accuracy:
table(pred.g.tuned, test_set[,5])
mean(pred.g.tuned == test_set[,5]) # Rate of well classified!

# Plot Decision Boundaries:
decisionplot(model.g.tuned, test_set[,3:5], class = "Species", main = "SVD (RBF)")
plot(model.g.tuned, test_set[,3:5])
plot(model.g.tuned, training_set[,3:5])


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

svp <- ksvm(Species ~ Petal.Width + Petal.Length, data = training_set,
            type="C-svc", kernel=kfunction2(2, 1, 1, 0), C = 1, prob.model=TRUE)
svp

## get fitted values
fit.svp = fitted(svp)
## Test on the training set with probabilities as output
pred.svp = predict(svp, test_set, type = "response")

# Check accuracy:
# ... on test
table(pred.svp, test_set[,5])
mean(pred.svp == test_set[,5]) # Rate of well classified!
# ... on train
table(fit.svp, training_set[,5])
mean(fit.svp == training_set[,5]) # Rate of well classified!

