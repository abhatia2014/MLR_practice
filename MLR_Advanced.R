
library(mlr)


# Configuring MLR ---------------------------------------------------------

#1. Reducing the output on the console

rdesc=makeResampleDesc("Holdout")
r=resample("classif.multinom",iris.task,rdesc)

#the output for the learner and resample can be suppressed as follows

lrn=makeLearner("classif.multinom",config = list(show.learner.output=FALSE))
r=resample(lrn, iris.task,rdesc,show.info = FALSE)

#to suppress the output learner from all subsequent learners

configureMlr(show.learner.output = FALSE, show.info = FALSE)

r=resample("classif.multinom",iris.task,rdesc)


#assessing and resetting the configurations

#function getMlrOptions returns a list with the current configuration

getMlrOptions()

#to configure to default, call configureMlr with an empty argument list

configureMlr()

# Example : handling errors in a learning method

#when you dont want a large experiment to stop due to one error
#using the on.learner.error option of configureMlr

train("classif.qda",task=iris.task,subset=1:104)

#gives an error as the group is too small for qda

#we change the configureMlr options

configureMlr(on.learner.error = "warn")

mod=train("classif.qda",task=iris.task,subset=1:104)
mod

#as mod has failed, it is an object of class FailureModel

isFailureModel(mod)

#retrieve the error message
getFailureModelMsg(mod)

#if a model is a failure, the predictions and performance will be NAs


# Wrapped Learners (Wrappers) ---------------------------------------------


