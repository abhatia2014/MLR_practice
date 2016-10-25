
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
classif.learn
#another example regression GBM and specify hyper parameters

regr.lrn=makeLearner("regr.gbm",par.vals = list(n.trees=500,interaction.depth=3))
regr.lrn

#another example- K Means with 5 clusters

cluster.lrn=makeLearner("cluster.kmeans",centers=5)

#the naming convention for the make learner method is 
#classif.<R_method_name>

#Hyperparameter values can be specified either via the ... argument or as a list via par.vals.

#fix.factor.prediction=TRUE can solve the problem of test and training data not with the same factors

#accessing a learner

names(classif.learn)

#get the complete set of hyperparameters for classif.learn

classif.learn$par.set
#this gives the configured parameters
classif.learn$par.vals
cluster.lrn$par.vals
regr.lrn$par.set
regr.lrn$par.vals

#get the type of prediction

cluster.lrn$predict.type

#getHyperPars gets all the set parameteres for a learner

getHyperPars(cluster.lrn)

#getParamset gives all possible setting
getParamSet(cluster.lrn)
getParamSet(regr.lrn)

#we can also get a quick overview of all parameters for a learning method before makeLearning

getParamSet("classif.randomForest")
?randomForest

#listing all available learners

lrns=listLearners()
#list classifiers that can output probabilities

lrns2=listLearners("classif",properties = "prob")
lrns2
#list learners that can be applied to iris (ie. multiclass) and output probabilities

lrns3=listLearners(iris.task,properties = 'prob')


# Training a Learner ------------------------------------------------------

# means fitting a model to the training dataset

# We start with a classification example and perform a linear discriminant analysis on the iris data set.

#1. generate a task

task1=makeClassifTask(data= iris,target = "Species")

#2. generate learner

lrn1=makeLearner("classif.lda")

#3. train the learner

mod1=train(lrn1,task1)

mod1

#alternatively, could also use
mod1 = train("classif.lda", task1)
#returns a class that represents the fitted model

#the model can be assessed using the function getLearnerModel

getLearnerModel(mod1)

#in this example we cluster the Ruspini dataset that has 4 groups and 2 features

data(ruspini,package="cluster")
head(ruspini)
plot(y~x,ruspini)

#generate task

task.cluster=makeClusterTask(data=ruspini)

#generate the learner

clus.lrn=makeLearner("cluster.kmeans",centers=4)

#train the learnere

clus.trn=train(clus.lrn,task.cluster)
getLearnerModel(clus.trn)
clus.trn
names(clus.trn)
clus.trn$learner

clus.trn$time

#extract the fitted model

getLearnerModel(clus.trn)

#The subset argument of train takes a logical or integer vector that indicates which observations to use, for example if you want to split your data into a training and a test set or if you want to fit separate models to different subgroups in the data
?train

#let's fit a linear regression model to Boston Housing data

#get the number of observations

n=getTaskSize(regr.task)

#use 1/3rd of the data for training

train.set=sample(n,n/3)

#train the learner

boston.mod=train("regr.lm",regr.task,subset = train.set)
boston.mod

#assigning weights

#in the example of lung cancer, benign class is almost twice as frequent as class malignant

#in order to grant both classes equal importance, we can weight the examples according to the inverse 
#class frequencies as below

#calculate the observation weights

target=getTaskTargets(classiftask)
tab=as.numeric(table(target))
tab
wt=1/tab[target]
wt

#now build the training model and specify the weights

train("classif.rpart",task = classiftask,weights = wt)
#this deals with imbalanced classification problems

#one can also specify the weights in task, but those in train take precedence over the ones in tasks


# Predicting Outcomes for New Data ----------------------------------------

#prediction method is the same, however the new data is passed as
#1. either pass the Task via Task argument
#2. or pass the data frame via new data argument

#use the boston housing and fit a GBM model

n=getTaskSize(regr.task)
n
train.set=seq(1,n,by=2)
test.set=seq(2,n,by=2)

lrn=makeLearner("regr.gbm",n.trees=100)
regmod=train(lrn,regr.task,subset = train.set)
task.pred=predict(regmod,task=regr.task,subset = test.set)
task.pred
names(task.pred)
task.pred$task.desc
task.pred$data

#for unsupervised machine learning problems

#we cluster the IRIS dataset w/o the target variable

n=nrow(iris)
iris.train=iris[seq(1,n,by=2),-5]
iris.test=iris[seq(2,n,by=2),-5]
task=makeClusterTask(data=iris.train)
mod=train("cluster.kmeans",task)
newdata.pred=predict(mod,newdata = iris.test)
newdata.pred

#assessing the prediction

head(as.data.frame(task.pred))

#direct way to access the true and predicted values of the target variables is
head(getPredictionTruth(task.pred))

#to get the actual response

head(getPredictionResponse(task.pred))

#to get probabilities, use the function- getpredictionprobabilities

#using fuzzy cmeans clustering

lrn=makeLearner("cluster.cmeans",predict.type = "prob")
mod=train(lrn,clustertask) #clustertask created for the mtcars dataset
pred=predict(mod,task=clustertask)
head(getPredictionProbabilities(pred))

#using prediction probabilities for classification tasks

#linear discriminent analysis on the IRIS dataset

mod=train("classif.lda",task = iris.task)
pred=predict(mod,task=iris.task)
pred

#to get predicted posterior probabilities

lrn=makeLearner("classif.rpart",predict.type = "prob")
mod=train(lrn,iris.task)
pred=predict(mod,newdata = iris)
head(as.data.frame(pred))
pred
pred$data

conf.matrix=getConfMatrix(pred,relative=FALSE)

head(getPredictionProbabilities(pred))

#a confusion matrix can be obtained by the function calculateconfusionmatrix


confusionMatrix(pred$data$response,pred$data$Species)

#generating confusion matrix in mlr

#adjusting the threshold

# threshold to decide if the predicted class is positive
sonar.task
lrn=makeLearner('classif.rpart',predict.type = "prob")
library(mlr)
mod=train(lrn,task = sonar.task)

#get the label of the positive class
getTaskDescription(sonar.task)$positive

pred1=predict(mod,sonar.task)
pred1$threshold

#set the threshold value for the positive class

pred2=setThreshold(pred1,0.9)
pred2$threshold
pred2

head(getPredictionProbabilities(pred1))
head(getPredictionProbabilities(pred1,cl=c("M","R")))

#using it for multiclass function

lrn=makeLearner("classif.rpart",predict.type = "prob")
mod=train(lrn,iris.task)
pred=predict(mod,newdata = iris)
pred$threshold
table(as.data.frame(pred)$response)
pred=setThreshold(pred,c(setosa=0.01,versicolor=50,virginica=1))
pred$threshold
table(as.data.frame(pred)$response)


# Visualizing the prediction ----------------------------------------------

#the function plotlearnerprediction helps visualizing the predictions

#for classification-we get a scatter plot of 2 features
#symbols with white border indicate misclassified observations

