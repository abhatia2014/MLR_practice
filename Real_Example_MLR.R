library(mlr)

#let's look at some of the algorithms available for classification

listLearners("classif")[c("class","package")]

#get the data from DataHack

train=read.csv("train.csv",na.strings=c(""," ",NA))
test=read.csv("test.csv",na.strings = c(""," ",NA))

#Data Exploration

summarizeColumns(train)

summarizeColumns(test)
summary(train)

#check for presence in skewness in variables

hist(train$ApplicantIncome,breaks=300)
hist(train$CoapplicantIncome,breaks = 100)

#the distributions are right skewed

boxplot(train$ApplicantIncome)

#first let's change the class of credit history to factor

train$Credit_History=as.factor(train$Credit_History)
test$Credit_History=as.factor(test$Credit_History)

summary(train)
levels(train$Dependents)
#change the 4th level (3+) to 3 

levels(train$Dependents)[4]=3
levels(test$Dependents)[4]=3

train_id=train$Loan_ID
test_id=test$Loan_ID
train$Loan_ID=NULL
test$Loan_ID=NULL
#Missing value Imputations

#in this case we'll use basic mean and mode imputations to impute missing data
?impute
imp=impute(train,classes = list(factor=imputeMode(),integer=imputeMean()),dummy.classes = c("integer","factor"),
          dummy.type = "numeric")
imp1=impute(test,classes = list(factor=imputeMode(),integer=imputeMean()),dummy.classes = c("integer","factor"),
           dummy.type = "numeric")

imp_train=imp$data
imp_test=imp1$data

summarizeColumns(imp_train)
summary(imp_train)
summary(imp_test)

#let's see which algorithms work with missing values

listLearners("classif",check.packages = TRUE,properties = "missings")[c("class","package")]

#treating missing values using the ML algorithm rpart
names(train)
rpart_imp=impute(train,target="Loan_Status",
                 classes=list(numeric=imputeLearner(makeLearner("regr.rpart")),
                              factor=imputeLearner(makeLearner("classif.rpart"))),
                 dummy.classes = c("numeric","factor"),
                 dummy.type = "numeric")
rpart_train_imp=rpart_imp$data
summary(rpart_train_imp)
summary(imp_test)

#not a good method since there are still some NAs remaining after doing the rpart imputation

#feature Engineering

#consist of feature transformation and feature creation

#let's remove outliers from variables like Applicant Income, Loan Amount

summary(imp_train$ApplicantIncome)
summary(imp_train$LoanAmount)

#here we'll cap the large values and set them to a threshold value as shown below

cd=capLargeValues(imp_train,target="Loan_Status",cols = c("ApplicantIncome"),threshold = 40000)
summary(cd$ApplicantIncome)
cd=capLargeValues(cd,target = "Loan_Status",cols = c("CoapplicantIncome"), threshold=21000)
summary(imp_train$LoanAmount)
cd=capLargeValues(cd,target = "Loan_Status",cols = c("LoanAmount"), threshold=520)

#rename the training set
cd_train=cd

#add a dummy variable (loan_status) in the test data

imp_test$Loan_Status=sample(0:1,size=367,replace = T)
#perform the same caplarge values for the test data

cd=capLargeValues(imp_test,target="Loan_Status",cols = c("ApplicantIncome"),threshold = 33000)
summary(cd$ApplicantIncome)
cd=capLargeValues(cd,target = "Loan_Status",cols = c("CoapplicantIncome"), threshold=16000)
summary(imp_train$LoanAmount)
cd=capLargeValues(cd,target = "Loan_Status",cols = c("LoanAmount"), threshold=470)

#renaming test_data

cd_test=cd

#convert the dummy binary numeric variables to factor using dplyr package

str(cd_train)

library(dplyr)

cd_train1=cd_train%>%
  select(contains(".dummy"))%>%
  mutate_each(funs(as.factor))

str(cd_train1) 

cd_train=cd_train[,-c(13:19)]
#colbind the two datasets
cd_train=cbind(cd_train,cd_train1)
rm(cd_train1)
str(cd_train)

# do the same for cd_test data 

cd_test1=cd_test%>%
  select(contains(".dummy"))%>%
  mutate_each(funs(as.factor))

str(cd_test1) 

cd_test=cd_test[,-c(13:18)]
#colbind the two datasets
cd_test=cbind(cd_test,cd_test1)
rm(cd_test1)
str(cd_test)

#we'll create some features now
#total income
cd_train$Total_Income=cd_train$ApplicantIncome+cd_train$CoapplicantIncome
cd_test$Total_Income=cd_test$ApplicantIncome+cd_test$CoapplicantIncome

#Income by Loan

cd_train$Income_by_loan=cd_train$Total_Income/cd_train$LoanAmount
cd_test$Income_by_loan=cd_test$Total_Income/cd_test$LoanAmount

#Loan amount by term

cd_train$Loan_amount_by_term=cd_train$LoanAmount/cd_train$Loan_Amount_Term
cd_test$Loan_amount_by_term=cd_test$LoanAmount/cd_test$Loan_Amount_Term

#find out variables with high correlations

#first load the caret package

library(caret)

#first split the data based on class

az=split(names(cd_train),sapply(cd_train,function(x) {class(x)}))
az$factor
az$numeric

#only taking the numeric ones out

xs=cd_train[az$numeric]

#now check correlation
cor(xs)
#very high correlation of applicant income with total income, 
#we can drop total _income
cd_train$Total_Income=NULL
cd_test$Total_Income=NULL

#alternatively find correlations that meet a specified criteria
xscorr=cor(xs)
highcorr=findCorrelation(xscorr,cutoff = 0.75)
highcorr
names(xs)
#we'll remove the column with the high correlation respresented by highcorr

#Machine Learning Tasks

#1. create a task

traintask=makeClassifTask(data = cd_train,target = "Loan_Status")
#add a dummy variable to cd_test
cd_test$Loan_Status=sample(0:1,size=367,replace = T)
#remove the extra dummy variable- Gender.dummy
cd_test=cd_test[,-13]
testtask=makeClassifTask(data=cd_test,target = "Loan_Status")

traintask

#change the positive class from N to Y

traintask=makeClassifTask(data=cd_train,target = 'Loan_Status',positive = "Y")
traintask

#for a deeper view of the traintask , 
str(getTaskData(traintask))

#Now we'll normalize the features using normalizeFeatures function- only numeric variables are normalized

traintask=normalizeFeatures(traintask,method = "standardize")
testtask=normalizeFeatures(testtask,method="standardize")

#we drop features that are not needed- married dummy

traintask=dropFeatures(traintask,features = c("Married.dummy"))

#Let's see which variables are important

#mlr has a built in function for feature importance

#install FSelector package



im_features=generateFilterValuesData(traintask,method = c("information.gain","chi.squared"))

