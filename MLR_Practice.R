
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

