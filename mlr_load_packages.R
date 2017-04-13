library(mlr)
data("iris")
head(iris)
task=makeClassifTask(data=iris,target = 'Species')

#find all classification learners

allclasslearners=listLearners(task)
allclassbenchmark=allclasslearners$class
allclassbenchmark=allclassbenchmark[-c(18,19,37,38,50)]
allclassbenchmark=allclassbenchmark[-17]

classbench=benchmark(learners = allclassbenchmark,tasks = task,resamplings = resampling)
plotclass=as.data.frame(classbench)
plotclass=plotclass %>% 
  arrange(mmce)
ggplot(plotclass,aes(x = reorder(learner.id,mmce),mmce,fill=learner.id))+geom_bar(stat="identity")+
  coord_flip()+geom_label(aes(label=mmce),size=2)+theme(legend.position = "none")

newdataframe=data.frame(counts=rnorm(100,10,2),morecounts=rnorm(100,65,8))
newdataframe$result=newdataframe$counts/rnorm(100)*newdataframe$morecounts
regrtask=makeRegrTask(data = newdataframe,target = 'result')
allregrlearners=listLearners(regrtask)

allmodels=allregrlearners$class
resampling=makeResampleDesc(method = "Holdout")

allmodels=allmodels[-13]
allmodels=allmodels[-c(28,49)]
benchmarkregr=benchmark(learners = allmodels,tasks = regrtask,resamplings = resampling,measures = mse)
plotdata=as.data.frame(benchmarkregr)
library(dplyr)
plotdata=plotdata %>% 
  arrange(mse)
library(ggplot2)
ggplot(plotdata,aes(x=reorder(learner.id,mse),mse,fill=learner.id))+geom_bar(stat="identity")+theme(legend.position = "none")+
  coord_flip()+geom_label(aes(learner.id,mse,label=round(mse,4)),size=2)

plotBMRBoxplots(benchmarkregr)

  