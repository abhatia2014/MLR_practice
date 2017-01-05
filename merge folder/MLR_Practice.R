
#Start the tutorial

# a simple example of stratified crossvalidation of LDA with R

#install.packages

library(mlr)
data(iris)
head(iris)

#1. define the task

task=makeClassifTask(id="tutorial",data=iris,target = "Species")
task$task.desc

#2. Define the learner

lrn=makeLearner("classif.lda")

#3. Define the resampling strategy

rdesc=makeResampleDesc(method="CV",stratify = TRUE)

?makeResampleDesc

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
classiftask$task.desc
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
?makeSurvTask
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
clustertask$task.desc

# Cost Sensitive Classification Task --------------------------------------

library(dplyr)
df=tbl_df(df)
df
iris=tbl_df(iris)
iris

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

# New Function- remove features with zero variance ------------------------

#add a constant feature to Breast Cancer task sampled at 5 places 
t=sample(nrow(df2),2,replace = FALSE)
BreastCancer$const=1
df2=BreastCancer
df2$const[t]=5


df2$Id=NULL
classiftask2=makeClassifTask(data = df2,target = "Class")
removeConstantFeatures(clustertask)
?removeConstantFeatures
classiftask2
removeConstantFeatures(classiftask2,perc = 0.05)
#removes a constant column const only when there is a percentage specified , like in the
#above case- any feature where less than 0.05 observations differ from the mode - 
#will be deleted
#remove selected features
#first get names
getTaskFeatureNames(surv.task)

# New Function-Can be used to drop selective features from the task --------------------


dropFeatures(surv.task,c("meal.cal","wt.loss"))


# New Function- normalizeFeatures -----------------------------------------

# for normalizing- center, scale, standardize, range
?normalizeFeatures

task=normalizeFeatures(clustertask,method = "range")
#to get a summary of the task data that has been normalized
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
library(rgl)
library(clusterSim)
cluster.lrn=makeLearner("cluster.kmeans",centers=5)

#the naming convention for the make learner method is 
#classif.<R_method_name>

#Hyperparameter values can be specified either via the ... argument or as a list via par.vals.

#fix.factor.prediction=TRUE can solve the problem of test and training data not with the same factors

#accessing a learner

names(classif.learn)
classif.learn$par.set
#get the complete set of hyperparameters for classif.learn

classif.learn$par.set

#this gives the configured parameters
classif.learn$par.vals

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
mod1$learner.model
mod1$features

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

# To Start from here >>> --------------------------------------------------


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

n=getTaskSize(bh.task)
n
train.set=seq(1,n,by=2)
test.set=seq(2,n,by=2)

lrn=makeLearner("regr.gbm",n.trees=100)
regmod=train(lrn,bh.task,subset = train.set)
task.pred=predict(regmod,task=bh.task,subset = test.set)
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
#using a confusion matrix

calculateConfusionMatrix(pred)

#get relative frequencies additional to the absolute numbers

conf.matrix=calculateConfusionMatrix(pred,relative=TRUE)
conf.matrix

conf.matrix$relative.row

#we can also add the absolute number of observations for each predicted and true class
#label to the matrix

calculateConfusionMatrix(pred,relative = TRUE,sums = TRUE)


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

#for classification-we get a scatter plot of 2 features (by default first two in the dataset)
#symbols with white border indicate misclassified observations

lrn=makeLearner("classif.rpart",id="CART")
plotLearnerPrediction(lrn,task = iris.task)

#plotting for clustering

#here color represents the predicted cluster
library(clusterSim)
lrn=makeLearner("cluster.kmeans")
plotLearnerPrediction(lrn,task=mtcars.task,features = c("disp","drat"),cv=0)

#for regression - there are two types of plots
#1. 1 D plots showing target value in relation to single feature

plotLearnerPrediction("regr.lm",features = "lstat",task=bh.task)

#2 D plots

plotLearnerPrediction("regr.lm",features=c("lstat","rm"),task=bh.task)


# Evaluating Learner Performance ------------------------------------------

#the function to call is performance on the prediction results

#typical performance measure for classification is mmce (misclassification error),
# accuracy (acc), or ROC analysis

#for regression , the mse (mean squared error) or mae (mean of absolute error)
# is a good peformance measure

#for clustering, the Dunn index (Dunn) is a good peformance indicator

# a list of all performance measures for classification of multiple classes

listMeasures("classif",properties = "classif.multi")

#performance measures suitable for iris classification tasks

listMeasures(iris.task)

#this function gives the default measure for the learning problem

getDefaultMeasure(iris.task)
getDefaultMeasure(bh.task)

#we'll fit a gradient boosing machine method on the boston housing data and calculate the default mse
bh.task
n=getTaskSize(bh.task)
lrn=makeLearner("regr.gbm",n.trees=1000)
mod=train(lrn,task = bh.task,subset = seq(1,n,by=2))
pred=predict(mod,task = bh.task,subset = seq(2,n,by=2))

performance(pred)

#if we want to know the median of squared error

performance(pred,measures = medse)

#we can also get multiple performance measures at once
performance(pred, measures=list(mse, medse,mae))

#for some performance measures like time to train, we have to pass the task or fitted model

performance(pred,measures = list(timetrain,timeboth,timepredict),model = mod)

#for many performance measures in clustering, the task is required

lrn=makeLearner("cluster.kmeans",centers=3)
mod=train(lrn, mtcars.task)
pred=predict(mod,task=mtcars.task)

performance(pred, measures=dunn,task=mtcars.task)

#Calculation of AUC for binary classification

lrn=makeLearner("classif.rpart",predict.type = "prob")
mod=train(lrn,task = sonar.task)
pred=predict(mod,task=sonar.task)

performance(pred,measures = list(mmce,auc))

#Access a performance measure

str(mmce)
str(auc)

#Binary classification

#we consider the sonar dataset

data("sonar.task")
sonar.task
lrn=makeLearner("classif.lda",predict.type = "prob")
n=getTaskSize(sonar.task)
mod=train(lrn,task = sonar.task,subset=seq(1,n,by=2))
pred=predict(mod,task=sonar.task,subset=seq(2,n,by=2))

#performance for the default threshold of 0.5

performance(pred,measures = list(fpr,fnr,mmce))

#lets plot the false positive and negative rate against the threshold

d=generateThreshVsPerfData(pred,measures = list(fpr,tpr,mmce))
plotROCCurves(d)
performance(pred,auc)
plotThreshVsPerf(d)

#lets plot using ggvis package

plotThreshVsPerfGGVIS(d)

#calculating performance using ROC measures
#function calculateROCMeasures

r=calculateROCMeasures(pred)
print(r,abbreviations = FALSE)


# Resampling --------------------------------------------------------------

#various resampling strategies- cross validation and bootstrap

#function to chose the strategy- makeResampleDesc

#different strategies
# Cross validation ("cv")
# Leave One out cross validation ("LOO")
# repeated cross validation ("RepCV")
# out of bag bootstrap and other variants ("Bootstrap")
# subsampling - called Monte Carlo Validation ("Subsample")
# Holdout ("Holdout")

#resample function evaluates the performance of a learner

#resampling strategy example (3 fold cross validation)

rdesc=makeResampleDesc("CV",iters=3)
rdesc
r=resample("surv.coxph",lung.task,rdesc)

r
names(r)
#gives the aggregated performance
r$aggr
#gives the performance of the test set
r$measures.test
r$measures.train

r$pred

#cluster iris feature data

task=makeClusterTask(data=iris[,-5])

#subsampling with 5 iterations and default split 2/3

rdesc=makeResampleDesc("Subsample",iters=5)

#another example with 5 iterations and 4/5 of the training data split

rdesc=makeResampleDesc("Subsample",iters=5, split=4/5)

#calculate the performance measures
library(clusterSim)
r=resample("cluster.kmeans",task,rdesc,measures=list(dunn,db,timetrain))

#Stratified Sampling

#stratified sampling ensures that all samples have the same proportion of classes 
#as the original data- particularly for imbalanced data

#put stratify=TRUE in the makeResampleDesc

# 3 fold cv

rdesc=makeResampleDesc("CV",iters=10,stratify = TRUE)

rdesc
r=resample("classif.lda",iris.task,rdesc)
r$pred

#stratification for survival tasks

rdesc=makeResampleDesc("CV",iters=10,stratify.cols = "chas")

r=resample("regr.rpart",bh.task,rdesc)

#accessing individual learner models

#for returning the models that are build by resample

rdesc=makeResampleDesc("CV",iters=3)

r=resample("classif.lda",iris.task,rdesc,models = TRUE)

r$models

#getting the variable importance in a regression tree using the extract argument

rdesc=makeResampleDesc('CV',iters=3)
r=resample('regr.rpart',bh.task,rdesc,extract = function(x) x$learner.model$variable.importance)

r$extract

str(rdesc)

#if you want to compare learner models, better to use makeResampleInstance

#a resample instance can be created from a function makeResampleInstance given a resampledesc
#and the size of data or the task
#makeresampleinstance performs random drawing of indices to separate the data into training and test set

rin=makeResampleInstance(rdesc,task=iris.task)
rin
rin$train.inds
rin$size
rin$test.inds
rin
rin=makeResampleInstance(rdesc,size=nrow(iris))
rin
str(rin)
rin$train.inds[[3]]
#as the indices are defined, carrying out the same experiment for different learners using the same
#indices can be a good comparison

#calculate the performance of two learners based on the same resampling
rdesc=makeResampleDesc("CV",iters=3)
rin=makeResampleInstance(rdesc,task=iris.task)
r.lda=resample("classif.lda",iris.task,rin,show.info = TRUE)
r.rpart=resample("classif.rpart",iris.task,rin,show.info = TRUE)
r.lda$aggr
r.rpart$aggr

#aggregating performance values

mmce$properties
mmce$aggr
rmse$aggr

#using different performance measures and aggregation

m1=mmce
m2=setAggregation(tpr,test.median)

rdesc=makeResampleDesc("CV",iters=10)
r=resample("classif.rpart",sonar.task,rdesc,measures = list(m1,m2))

#Example - calculating the training error

#we have to set predict="both" to get the predictions on both training and test set

mmce.train.mean=setAggregation(mmce,train.mean)
rdesc=makeResampleDesc("CV",iters=10,predict = "both")
r=resample("classif.rpart",iris.task,rdesc,measures=list(mmce.train.mean,mmce))
r$measures.train
r$measures.test
r$aggr


#example of Bootstrap -draw with replacement
rdesc=makeResampleDesc("Bootstrap",predict = "both",iters=10)
listMeasures(iris.task)
r=resample('classif.rpart',iris.task,rdesc,measures = list(mmce,acc))
r$aggr
r$measures.test

#convenience functions
#some frequently used resampling strategies that can be used w/o much inconvenience

holdout("regr.lm",bh.task,measures = list(mse,mae))
crossval('classif.lda',iris.task,iters=3,measures = list(mmce,ber))


# Tuning Hyperparameters --------------------------------------------------

#in order to tune a machine learning algorithm, you have to specify
#1. the search space
#2. the optimization algorithm (tuning method)
#3. the evaluation method (resampling method and performance measure)

#an example of search space

#example create a search space for searching the values of c parameter of SVM between 0.01 to 0.1

getParamSet("classif.ksvm")
?ksvm
ps=makeParamSet(
  makeNumericParam("C",lower = 0.01, upper= 0.1)
)

#an example of the optimization algorithm could be performing random search on in the space
#random search with 100 iterations

ctrl=makeTuneControlRandom(maxit = 100L)

#an example of evaluation method could be a 3 fold CV using accuracy as the performance measure

rdesc=makeResampleDesc("CV",iters=3)
measure=acc

#doing in more details

#in this example well use the iris classification using SVM function
#here we will tune the c (cost) parameter and sigma of ksvm function

#specifying the search space (more details)

#first we give discrete values to the parameters

discrete_ps=makeParamSet(
  makeDiscreteParam("C",values = c(0.5,1.0,1.5,2.0)),
  makeDiscreteParam("sigma",values=c(0.5,1.0,1.5,2.0))
)

discrete_ps

#we could also define a continous search space using makeNumericParam
#in the search space 10^-10 to 10^10

# use trafo argument for transformation

num_ps=makeParamSet(
  makeNumericParam("C",lower=-10, upper=10, trafo = function(x) 10^x),
  makeNumericParam("sigma",lower=-10, upper=10, trafo = function(x) 10^x)
)

num_ps

#specifying the optimiation algorithm (detailed)

#a grid search is one of the standard- though slower

#in the case of discrete search , the grid search will be a cross product

ctrl=makeTuneControlGrid()

#in case of num_ps (numeric search parameter) , grid search will create a grid using equally sized steps (10)

#the default 10 steps can be changed specifying 
ctrl = makeTuneControlGrid(resolution = 15)

#if the grid search is too slow, we can use random search, it will randomly choose from the specified values

#the maxit arguement controls the number of iterations

ctrl=makeTuneControlRandom(maxit = 10L)

#in the case of num_ps, random search will randomly choose points in the specified bounds

ctrl=makeTuneControlRandom(maxit = 200)

#performing the tuning

#first define a sampling strategy and peformance measure

rdesc=makeResampleDesc("CV",iters=3)

#finally we combine all the previous pieces
ctrl=makeTuneControlGrid()
res=tuneParams("classif.ksvm",task = iris.task,resampling = rdesc,
               par.set = discrete_ps,control = ctrl,measure=acc)
res

#tune param performs cross validation for every element of the cross product and selects the 
#hyperparameter with the best performance

#let's again tune parameters using the numeric parameter set

#search space
num_ps=makeParamSet(
  makeNumericParam("C",lower=-10,upper=10, trafo = function(x) 10^x),
  makeNumericParam("sigma",lower=-10,upper =10, trafo = function(x) 10^x)
)

#optimization algorithm

ctrl=makeTuneControlRandom(maxit = 100)

#tune parameters

res=tuneParams("classif.ksvm",task=iris.task,resampling = rdesc, par.set = num_ps,
              control = ctrl,measures = list(acc,setAggregation(acc,test.sd)))
res

#accessing the tuning results

res$x

res$y

#we can now create a learner with optimal hyperparameter settings as follows

lrn=setHyperPars(makeLearner("classif.ksvm"),par.vals = res$x)

lrn
#now we train the model using the learner on the complete iris dataset

m=train(lrn,iris.task)
pred=predict(m,task=iris.task)
calculateConfusionMatrix(pred)

#investigating hyperparameter tuning effects

#we can investigate all points evaluated during the search

generateHyperParsEffectData(res)

#in order to get the actual parameter on the transformed scale, use trafo argument

generateHyperParsEffectData(res,trafo = TRUE)

#here we generate performance on both the test and train data using predict="both" argument

rdesc2=makeResampleDesc("Holdout",predict="both")

#and tune the parameters

res2=tuneParams("classif.ksvm",task=iris.task,resampling = rdesc2,control = ctrl,
                par.set = num_ps,measures = list(acc,setAggregation(acc,train.mean)))
res2
res2$y

#we'll now visualize the points using the plothyperparseffect
res=tuneParams("classif.ksvm",task=iris.task,resampling = rdesc, par.set = num_ps,
               control = ctrl,measures = list(acc,mmce))
res
data=generateHyperParsEffectData(res)
plotHyperParsEffect(data,x="iteration",y="acc.test.mean",plot.type = "line")

#do the same with some other classifier, let's say randomForest

getParamSet("classif.randomForest")
?randomForest
#we'll take 3 parameters ntree, mtry and sampsize

#we use  a search based on discrete values

search1=makeParamSet(
  makeDiscreteParam("ntree",values = c(100,300,500,700,1000,3000)),
  makeDiscreteParam('mtry',values = c(1,2,3,4,5)),
  makeDiscreteParam("sampsize",values = c(10,20,30))
)

#define the optimization algorithm

ctrl1=makeTuneControlGrid(resolution = 20)

#define the sampling and performance measures

resampleIris=makeResampleDesc("CV",iters=10)
measures=list(acc,mmce)

#tune the parameters

mytune=tuneParams("classif.randomForest",task = iris.task,resampling = resampleIris,
                  par.set = search1,measures = measures,control = ctrl1)
mytune

#so the optimized parameter set is ntree=700, mtry=2, sampsize=10

#we use these parameters to create the learner model using sethyperpars


irislearner=setHyperPars(makeLearner("classif.randomForest"),par.vals = mytune$x)

#train the model

iristrain=train(irislearner,task = iris.task)

#perform the prediction

irispred=predict(iristrain,task=iris.task)

calculateConfusionMatrix(irispred) #using randomForest model
calculateConfusionMatrix(pred) #using ksvm model


# Benchmarking Experiments ------------------------------------------------

# in a benchmarking experiment, different learning methods are applied to one or several
#dataset with the aim to compare and rank the algorithms with respect to the performance measures

#done by calling a function 'benchmark' on a list of learners and a list of tasks

#benchmark executes a resample for each learner and task

#example on two learners - LDA (Linear Discriminant Analysis) and Classification Tree (rpart)
#applied to sonar.task

#resampling strategy - holdout

lrns=list(makeLearner("classif.lda",predict.type = "prob"),makeLearner("classif.rpart",predict.type = "prob"),makeLearner("classif.randomForest",predict.type = "prob"),makeLearner("classif.ksvm",predict.type = "prob"))

#chosing the resampling strategy

rdesc=makeResampleDesc("Holdout")

#conduct the benchmark experiment

listMeasures(sonar.task)

bmr=benchmark(lrns,sonar.task,rdesc,measures = list(mmce,acc,tpr,fpr,auc,kappa))

bmr

bmr$measures

#accessing benchmark results

#getBMRPerformances returns individual performances in the resampling runs
#whitle getBMRAggrPerformances gets aggregated values

getBMRPerformances(bmr)

getBMRAggrPerformances(bmr)

#to convert it into a dataframe, put as.df=TRUE

t=getBMRPerformances(bmr,as.df = TRUE)

#predicting results

#assess predictions using getBRMPredictions

getBMRPredictions(bmr)

#to access results of a certain learner by their ID

head(getBMRPredictions(bmr,learner.ids = "classif.randomForest",as.df = TRUE))

#get the IDs of all learners or tasks

getBMRTaskIds(bmr)
getBMRLearnerIds(bmr)

#also measures IDs
getBMRMeasureIds(bmr)

#the fitted models can be retrieved using the syntax getBMRModels

getBMRModels(bmr)

#extracting the learners

getBMRLearners(bmr)

#extract measures

getBMRMeasures(bmr)

#Merging benchmark results

#merging two benchmark results using mergeBenchmarkResultLearner

#first benchmark result as a dataframe

bmr

#define second benchmark
listLearners(sonar.task)
lrns2=list(makeLearner("classif.gbm",predict.type = "prob"),makeLearner("classif.avNNet",predict.type = "prob"),
           makeLearner("classif.C50",predict.type = "prob"))

bmr2=benchmark(lrns2,sonar.task,rdesc,measures = list(mmce,acc,tpr,fpr,auc,kappa))

bmr2

#now merge the results of bmr2 with bmr

bmrfinal=mergeBenchmarkResultLearner(bmr,bmr2)
getBMRPredictions(bmrfinal,as.df = TRUE)
getBMRPerformances(bmrfinal,as.df = TRUE)

#in the case of BMR and BMR2 , different resamples were passed to the benchmarks
#better to work with resampleinstances from the begining

#alternatively, one can extract the resampleInstances from the first benchmark experiment
# and pass to all future benchmark calls

rin=getBMRPredictions(bmr)[[1]][[1]]$instance

rin
bmr3=benchmark(lrns2,sonar.task,rin,measures = list(mmce,acc,tpr,fpr,auc,kappa))

#now merge the bmr and bmr3

bmr_sample_final=mergeBenchmarkResultLearner(bmr,bmr3)
bmr_sample_final

#Benchmark analysis and visualization

#analyzing the results of the benchmarking experiment by 
#visualization, algorithm ranking, and hypothesis testing

#we conduct a larger benchmark experiment with three learning experiment applied to five classification tasks

#example comparing lda, rpart, randomForest and ksvm

#we'll choose 5 tasks 
#and use 10 fold CV as the resampling strategy

#use mmce are the primary performance measure, also calculate ber (balanced error rate) and training time (timetrain)

lrns=list(
  makeLearner("classif.lda",id='lda'),
  makeLearner("classif.rpart",id='rpart'),
  makeLearner("classif.randomForest",id="randomForest"),
  makeLearner("classif.ksvm",id="svm"),
  makeLearner("classif.gbm",id="gbm")
)

#get additional tasks from the package mlbench

ring.task=convertMLBenchObjToTask("mlbench.ringnorm",n=600)
wave.task=convertMLBenchObjToTask("mlbench.waveform",n=600)

#define all tasks

tasks=list(iris.task,sonar.task,pid.task,ring.task,wave.task)

#define the resampling strategy

rdesc=makeResampleDesc("CV",iters=10)

#define measures

meas=list(mmce,ber,timetrain)

#run the benchmark experiment

bmr=benchmark(lrns,tasks,rdesc,meas)
bmr

#the individual performance on the 10 folds for every task, learner and measure can be 
#retrieved as below

perf=getBMRPerformances(bmr,as.df = TRUE)
perf

aggrperf=getBMRAggrPerformances(bmr,as.df = TRUE)
aggrperf

#integrated Plots

#visualizing performance
#plotBMRBoxPlot create box or violin plots

plotBMRBoxplots(bmr,measure = mmce)

#or as violin plots

plotBMRBoxplots(bmr,measure = ber,style = "violin")+
  aes(color=learner.id)+theme(strip.text.x=element_text(size=8))

mmce$name #gives the name of the measure ie. mean misclassification error

#changing the panel header names and learner names (xaxis)

plt=plotBMRBoxplots(bmr,measure=mmce,style="violin")
levels(plt$data$task.id)=c('Iris',"Ringnorm","Waveform","Diabetes","Sonar")
levels(plt$data$learner.id)=c("lda","rpart","rf","svm","gbm")
plt+aes(color=learner.id)

#visualizing aggreageted performance using plotBMRSummary

plotBMRSummary(bmr)

#calcualating and visualizing ranks

#function convertBMRToRankMatrix calculates ranks based on aggregated learning performance
#of one measure, say mmce

m=convertBMRToRankMatrix(bmr,mmce)
m
#the rank structure can be visualized as a bar chart

plotBMRRanksAsBarChart(bmr,mmce)

#plotBMRanksAsBarChart shows it as a heatmap

plotBMRRanksAsBarChart(bmr,pos = "tile",mmce)

#also plotting the summary with ranks as trafo

plotBMRSummary(bmr,trafo = "rank")

#comparing learners using hypothesis tests

#we use the overall Friedman test and Friedman-Nemenyi post hoc test

friedmanTestBMR(bmr)

#since the pvalue is 0.008, we can reject the null hypothesis with the alternative hypothesis that
#the difference between the learners is significant

#we want to find where this difference lies - we use friedmanposthoctest

friedmanPostHocTestBMR(bmr,p.value = 0.05)

#critical differences diagram

#differnces can be plotted using (test='bd') or (test='nemenyi')

#nemenyi test

g=generateCritDifferencesData(bmr,p.value = 0.05,test="nemenyi")
plotCritDifferences(g)


#Bonferroni-Dunn Test

g=generateCritDifferencesData(bmr,p.value = 0.05,test='bd')
plotCritDifferences(g)

#Custom plots

#creating density plots

perf=getBMRPerformances(bmr,as.df = TRUE)

ggplot(perf,aes(mmce,color=learner.id))+geom_density()+facet_wrap(~task.id)


# Parallelization ---------------------------------------------------------

#parallel map supports mlr
#good to call parallelstop at the end of the script

#loading parallelmap library

library(parallelMap)

#starting parrellization in mode=socket with cpus=1

parallelStartSocket(2)

rdesc=makeResampleDesc("CV",iters=10)

r=resample("classif.lda",iris.task,rdesc)

parallelStop()
#this stops the parallelization

#the levels that are supported for parallelization

parallelGetRegisteredLevels()

#one can only do parallel computing on a level and not on the whole by 
#passing 'mlr.resample' to the parallelStart function


# Visualization -----------------------------------------------------------

#visualization is performed by using the 'generation' function

#all generation function start with 'generate' followed by the function purpose
#plotting functions are prefixed by 'plot'

#examples
lrn=makeLearner("classif.lda",predict.type = "prob")
n=getTaskSize(sonar.task)

mod=train(lrn,sonar.task,subset = seq(1,n,by=2))
pred=predict(mod,sonar.task,subset = seq(2,n,by=2))
d=generateThreshVsPerfData(pred,measures = list(fpr,fnr,mmce))

class(d)
d$data
plotThreshVsPerf(d)

#manipulating the ggplot object

#changing the panel name for meam misclassification error to error rate

plt=plotThreshVsPerf(d)
head(plt$data)
levels(plt$data$measure)
plt$data$measure=factor(plt$data$measure,levels=c("mmce","fpr","fnr"),
                        labels=c("Eror Rate","FP Rate","FN Rate"))
plt

#alernatively, we caould manually create plots using ggplot2
head(d$data)
ggplot(d$data,aes(threshold,fpr))+geom_line()
