library(rJava)
library(coreNLP)
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
?makePreprocWrapperCaret
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
# from 41% to 24%

#joint tuning of preprocessing options and learner parameters

#the preprocessing and learner parameters can be tuned jointly

#let's first get an overview of all parameters using the function getParamSet

getParamSet(lrn)

#let's tune the number of principal components (ppc.pcaComp)

#we perform a grid search and set the resolution to 10

ps=makeParamSet(
  makeIntegerParam("ppc.pcaComp",lower=1, upper= getTaskNFeats(sonar.task)),
  makeDiscreteParam("predict.method",values=c("plug-in",'debiased'))
)

ctrl=makeTuneControlGrid(resolution=10)

res=tuneParams(lrn,sonar.task,rin,par.set = ps,control = ctrl)
res
as.data.frame(res$opt.path)[1:3]
summarizeColumns(mtcars)

summarizeColumns(iris)


# Imputation of Missing Values --------------------------------------------


#can impute by a fixed constant- mean , median or mode

#some algorithms can deal with missing values - obtained by list learners with properties- missing

listLearners("regr",properties = "missings")[c("class","package")]

#lets look at the airquality dataset
data("airquality")

summary(airquality)

#we'll insert some artificial NA in column wind and coerce it into a factor

airq=airquality
ind=sample(nrow(airq),10)

ind
airq$Wind[ind]=NA

airq$Wind=cut(airq$Wind,c(0,8,16,24))
summary(airq)
#we can impute Ozone and Solar.R missing values by mean and Wind missing values
#by mode and put dummy variables for the features that have missing values

str(airq)

imp=impute(airq,classes = list(integer=imputeMean(),factor=imputeMode()),
           dummy.classes = "integer")
head(imp$data)

imp$desc
summary(imp$data)

#let's look at another example involving a target variable
#here we will predict the ozone levels, first remove day and month variable
names(airq)
airq=airq[,c(1:4)]

#use first 100 as training and remaining as test set

airq.train=airq[1:100,]
airq.test=airq[-c(1:100),]

#for supervised learning, we need to pass the name of the target variable to impute
#prevents creation of a dummy variable for the target variables

#here we'll specify imputation method for each variable

#missing values for Solar.R are imputed by random numbers drawn from the empirical distribution
#of non missing numbers

#imputelearner allows us to use supervised learning algorithm integrated into mlr for imputation

#missing values in wind are replaced by predictions of a classification tree (rpart)

#rpart can deal with missing values therefore NA in solar.R do not pose a problem

imp=impute(airq.train,target = "Ozone",cols = list(Solar.R=imputeHist(),
                                                   Wind=imputeLearner("classif.rpart")),
           dummy.cols = c("Solar.R","Wind"))
summary(imp$data)
imp$desc

#the impdesc object can be used to reimpute the test data in the same way as the training data

airq.test.imp=reimpute(airq.test,imp$desc)
head(airq.test.imp)

#evaluating a machine learning method by resampling technique, the impute/reimpute
# may be called automatically each time before training/prediction- using an imputation wrapper

#example

lrn=makeImputeWrapper("regr.lm",cols = list(Solar.R=imputeHist(),
                                            Wind=imputeLearner("classif.rpart")),
                      dummy.cols = c('Solar.R','Wind'))
lrn

#imputelearner (lrn) is applied on the training set before the training
#then reimpute is applied on the test set and then predictions are done

#in this example, first we delete the missing values in the target variable
#before assigning the task

airq=airq[!is.na(airq$Ozone),]

#now create a task
task=makeRegrTask(data=airq,target = 'Ozone')

#create the resampling distribution

rdesc=makeResampleDesc("CV",iters=3)

#perform the resample

r=resample(learner = lrn,resampling = rdesc,task = task,models = TRUE)
r$aggr
r$models
lapply(r$models,getLearnerModel,more.unwrap=TRUE)


# Generic Bagging ---------------------------------------------------------

#bag any mlr learner using makeBaggingWrapper

#bw.iters- how many samples do we want to train our learner on
#bw.replace- sample with replacement
#bw.size- percentage of samples
#bw.feats- percentage size of randomely selected features for each iteration

#first let's setup a learner to pass the makeBaggingWrapper

lrn=makeLearner("classif.rpart")

baglrn=makeBaggingWrapper(learner = lrn,bw.iters = 50, bw.replace = TRUE,bw.size = 0.8,bw.feats = 0.75)

#now let's compare performance with and without bagging

rdesc=makeResampleDesc("CV",iters=10)

r=resample(learner = baglrn,task = sonar.task,resampling = rdesc)
r$aggr
#test.mean for mmce is 19.09%

# now resampling without bagging

normallrn=resample(learner = lrn,task = sonar.task,resampling = rdesc)
normallrn$aggr
#mean mmce is 27.88%
#thus there is an improvement of ~8% with bagging

#changing the type of prediction

baglrn=setPredictType(baglrn,predict.type = "prob")

n=getTaskSize(bh.task)
train.inds=sample(n,size = 0.65*n,replace = FALSE)
test.inds=setdiff(1:n,train.inds)

lrn=makeLearner("regr.rpart")
baglrn=makeBaggingWrapper(lrn)
baglrn=setPredictType(baglrn,predict.type = "se")
mod=train(learner = baglrn,task = bh.task,subset = train.inds)
#get the model
head(getLearnerModel(mod))

#predict the response and calculate the standard deviation

pred=predict(mod,task=bh.task,subset = test.inds)

head(as.data.frame(pred))

#we'll visualize it using ggplot2

library(ggplot2)
library(reshape2)

data=cbind(as.data.frame(pred),getTaskData(bh.task,subset = test.inds))
g=ggplot(data,aes(x=lstat,y=response,ymin=response-se,ymax=response+se,col=age))
g+geom_point()+geom_linerange(alpha=0.5)



# Feature Selection -------------------------------------------------------

#MLR supports filter and wrapper methods for performing feature selection

#generateFilterValuesData is the method to get feature importance and built into the mlr's function

#involves a task and character string specifying the method
library(RWeka)
fv=generateFilterValuesData(iris.task,method = "information.gain")
##*******rJava not working- will complete this section later*************##



# ROC Analysis and Performance Curves -------------------------------------


