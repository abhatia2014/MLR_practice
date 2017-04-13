library(mlr)

#steps to do machine learning analysis in R


# Quick Start -------------------------------------------------------------


#0 make the sample set for training and testing

train.set=sample(nrow(iris),size = 2/3*nrow(iris))
test.set=setdiff(1:nrow(iris),train.set)

#1. define the task
task=makeClassifTask(data=iris,target = "Species")

#2 define the learner
lrn=makeLearner("classif.lda")

#3 fit the model
model=train(learner = lrn,task = task,subset = train.set)

#4 make predictions
pred=predict(model,task=task,subset = test.set)

#5 Evaluate the learner
#here calculate the mean misclassification error and accuracy

performance(pred = pred,measures = list(mmce,acc))



# Tasks -------------------------------------------------------------------

# to create task make<task type>

#makeclassiftask selects the first level as the positive class, to change that, you can manually select the other class

data(BreastCancer, package = "mlbench")
df=BreastCancer
str(df)
df$Id=NULL
classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "Class", positive = "malignant")

#To access a task

#gettaskdesc

getTaskDesc(classif.task)

getTaskData(classif.task)

getTaskType(classif.task)

getTaskTargetNames(classif.task)

#get number of observations

getTaskSize(classif.task)

#get number of input variables
getTaskNFeats(classif.task)

#get the class levels
getTaskClassLevels(classif.task)

#get the names of features
getTaskFeatureNames(classif.task)

head(getTaskTargets(classif.task),10)

#modifying an existing task

#remove constant features

removeConstantFeatures(classif.task)


#remove selected features
dropFeatures(classif.task,c("Marg.adhesion"))

#standardize numerical features

task=normalizeFeatures(classif.task,method="range")
summary(getTaskData(task))



# Learners ----------------------------------------------------------------

#make a learner and set hyperparameters via a list

getHyperPars(regr.lrn)

regr.lrn=makeLearner("regr.gbm",par.vals = list(n.trees=500,interaction.depth=3))

C50.lrn=makeLearner("classif.C50")

#Occasionally, factor features may cause problems when fewer levels are present in the test data set than in the training data. By setting fix.factors.prediction = TRUE these are avoided by adding a factor level for missing data in the test data set.

#accessing a learner

#get the defaults hyperparameters setting for the learner
regr.lrn$par.set
C50.lrn$par.set
#can also get it from the generic function getparamsset

getParamSet(regr.lrn)


#Get the configured hyperparameters settings that deviate from the defaults
regr.lrn$par.vals
#or using the generic function
getLearnerParVals(regr.lrn)

#get the type of prediction
regr.lrn$predict.type

#modifying a learner

#change the prediction type
classif.lrn=setPredictType(C50.lrn,"prob")

#change hyperparamater values
classif.lrn=setHyperPars(classif.lrn,trials=20)
classif.lrn$par.vals

#to go back to default hyperparameter values
back.lrn=removeHyperPars(classif.lrn,"trials")
back.lrn$par.vals

#listing learners based on certain criteria

#listing learners that can output probabilities

lrns=listLearners("classif",properties = "prob")
lrns[c("class","package")]


# Training a Learner ------------------------------------------------------

task = makeClassifTask(data = iris, target = "Species")
mod=train("classif.lda",task = task)
mod

#Accessing learner models that are trained

#the fitted model can be accessed by the function- get learner model

data(ruspini,package="cluster")
plot(y~x,ruspini)
ruspini.task=makeClusterTask(data = ruspini)
lrn=makeLearner("cluster.kmeans",centers=4)

mod=train(learner = lrn,task = ruspini.task)

mod
names(mod)
mod$learner
mod$features

#extract the fitted model
getLearnerModel(mod)


