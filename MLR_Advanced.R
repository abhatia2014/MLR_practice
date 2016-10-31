
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

#examples include data preprocessing, imputation, bagging, tuning, feature selection e.t.c

#Example Bagging Wrapper
#here we create a random Forest that supports weights
# to achieve it, we combine several decision trees from the rpart packages to create our 
#own customized random forest

#first we create a weighted toy task

data(iris)
task=makeClassifTask(data=iris,target="Species",weights = as.integer(iris$Species))
task

#we use makeBaggingWrapper to create base learner and bagged learner, ntree (100 base learner)
#and mtry (proportion of randomely selected features)

#first base learner
base.lrn=makeLearner("classif.rpart")

#next bagged learner

wrapped.lrn=makeBaggingWrapper(base.lrn,bw.iters = 100, bw.feats = 0.5)
wrapped.lrn

#this new wrapped.lrn can be used for training, benchmarking, resample ,e.t.c

benchmark(tasks = task, learners = list(base.lrn,wrapped.lrn))

#the new bagged learner we created done better than single rpart

#lets tune some hyperparameters of a fused learner

getParamSet(wrapped.lrn)

#let's tune the parameter minsplit and bw.feats using a random search 3 fold CV

ctrl=makeTuneControlRandom(maxit = 10)

rdesc=makeResampleDesc("CV",iters=3)

par.set=makeParamSet(
  makeIntegerParam("minsplit",lower = 1,upper = 10),
  makeIntegerParam("bw.feats",lower=0.25,upper = 1)
)

tuned.lrn=makeTuneWrapper(wrapped.lrn,rdesc,mmce,par.set,ctrl)

tuned.lrn

#calling the train method of the newly constructed learner

lrn=train(tuned.lrn,task=task)


# Data Preprocessing ------------------------------------------------------

#transformation of data before applying a learning algorithm

#resolving inconsistencies, imputation of missing values, identifying-removing outliers,
#discretizing numerical data or generating dummy variables, dimension reduction, feature extraction

#1. capLargeValues- convert large/infinite numeric values
#2. createDummyFeatures- generate dummy variables for factor variables
#3. dropFeatures- remove selected features
#4. joinClassLevels- merge existing classes into new classes
#5. mergeSmallFactorLevels- merge infrequent levels of factor features
#6. normalizeFeatures- standardization/ scaling
#7. removeConstantFeatures- remove constant features
#8. subsetTask- remove observations or features from a task

#Fusing learners with preprocessing

#makePreprocWrapperCaret is an interface to all preprocessing otions offered by caret's preprocess function

#makePreprocWrapper permits to write own custom preprocessing methods by defining actions to be taken 
#before preprocessing and prediction

#1. preprocessing using the Caret package
#makePreprocWrapperCaret takes the same preprocess functions as caret but their names
#are prefixed by ppc

#example for a dataframe, the preprocess call will be

makePreprocWrapperCaret(learner="classif.lda",ppc.knnImpute=TRUE, ppc.pca=TRUE, ppc.pcaComp=10)

#pca should be only applied when reducing dimensions
#let's consider sonar.task

sonar.task

#keeping a threshold of 0.9 i.e principle componenets needed to explain a cumulative percentage of 90%
# of the total variance are retained

lrn=makePreprocWrapperCaret("classif.qda",ppc.pca=TRUE,ppc.thresh=0.9)
lrn

#training the model

mod=train(lrn, sonar.task)
mod
getLearnerModel(mod)
#we see that the model is trained on 22 features as determined by pca

getLearnerModel(mod,more.unwrap = TRUE)

#now we see the performance of qda with and without pca
#we use stratified sampling to prevent errors in qda

rin=makeResampleInstance("CV",iters=3,stratify=TRUE,task=sonar.task)
res=benchmark(list("classif.qda",lrn),sonar.task,rin)
res
# preprocessing has turned out to be really beneficial by reducing the mmce 
# from 39% to 23%

