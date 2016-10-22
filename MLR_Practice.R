
#Start the tutorial

# a simple example of stratified crossvalidation of LDA with R

#install.packages
install.packages("mlr")
library(mlr)
data(iris)
head(iris)

#1. define the task

task=makeClassifTask(id="tutorial",data=iris,target = "Species")

#2. Define the learner

lrn=makeLearner("classif.lda")

#3. Define the resampling strategy

rdesc=makeResampleDesc(method="CV",stratify = TRUE)



#4. Do the resampling

r=resample(learner = lrn, task = task, resampling = rdesc, show.info = TRUE)

#5. get the mean misclassfication error

names(r)
r$pred
r$aggr


# Tutorial MLR ------------------------------------------------------------

#Learning Tasks

#RegrTask- for regression problems
#ClassifTask- for classification problems
#SurvTask- for survival task classification
#ClusterTask- for cluster analysis
#MultilabelTask- for Multilabel classification problems
#CostSensTask- for cost sensitive classification


# Regression Task Example -------------------------------------------------

data(BostonHousing,package = "mlbench")
head(BostonHousing)

regr.task=makeRegrTask(id="bh",data=BostonHousing,target = "medv")
regr.task


# Classification Machine Learning Task ------------------------------------

data(BreastCancer,package = "mlbench")
head(BreastCancer)

df=BreastCancer
str(df)
df$Id=NULL
classiftask=makeClassifTask(id="bc",data=df,target = "Class")
#in the makeClassifTask - the positive class can be set by positive="malignant"

classiftask


# Survival Analysis Tasks -------------------------------------------------

#Survival tasks use two target columns

install.packages("survival")
library(survival)
data(lung,package = "survival")
head(lung)
lung$status=(lung$status==2)#converts to logical

surv.task=makeSurvTask(id="ls",data = lung,target = c("time","status"))
surv.task


# Multilabel Classification Tasks -----------------------------------------


