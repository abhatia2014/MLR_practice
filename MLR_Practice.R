
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

#in multilabel classification, each object can belong to more than one category
#at one time

#we use the 'yeast' dataset

yeast=getTaskData(yeast.task)
labels=colnames(yeast)[1:14]
yeast.task=makeMultilabelTask(id="multi",data = yeast,target = labels)
yeast.task


# Cluster Analysis Task--------------------------------------------------------

#we perform the Cluster Analysis Task using the MT Cars Data

data("mtcars",package='datasets')
head(mtcars)
clustertask=makeClusterTask(data=mtcars)
clustertask


# Cost Sensitive Classification Task --------------------------------------

library(dplyr)
df=tbl_df(df)
df
iris=tbl_df(iris)
iris
?runif
cost = matrix(runif(150 * 3, 0, 2000), 150) * (1 - diag(3))[iris$Species,]
cost.sens.taks=makeCostSensTask(id="costsens",data=iris,cost=cost)
cost.sens.taks

#to get a task description, we use the following function

getTaskDescription(classiftask)

#to get the number of input variables

getTaskNFeats(classiftask)

#to get task levels

getTaskClassLevels(classiftask)

#get the names of input variables in a task

getTaskFeatureNames(classiftask)
getTaskFeatureNames(clustertask)

#removing features with zero variance (constant features)

removeConstantFeatures(clustertask)

removeConstantFeatures(classiftask)

#remove selected features
#first get names
getTaskFeatureNames(surv.task)

dropFeatures(surv.task,c("meal.cal","wt.loss"))

#standardize numerical features
?normalizeFeatures

task=normalizeFeatures(clustertask,method = "range")
summary(getTaskData(task))


# Constructing Learners ---------------------------------------------------

#makelearner is to specify which learner (method) algorithm to use 

#example classification trees 
?makeLearner
classif.learn=makeLearner('classif.randomForest',predict.type = "prob",fix.factors.prediction = TRUE)

