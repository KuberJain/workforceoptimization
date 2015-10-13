#Predictive Goals
#1. Production Delay - Department/Project Types
#2. Character mismatch - Department/Project Types
#3. Slack times - Department/Project Types
#4. Demand vs Supply utilization
 

### Data Splitting
set.seed(1)
inTrain=createDataPartition(orderDataStats$Diff.In.Char.Count,p=0.75,list=FALSE)
orderTrain=orderDataStats[inTrain,]
orderTest=orderDataStats[-inTrain,]

#Divide the entire dataset into 3 equal parts
set.seed(1234)
vehicles=vehicles[sample(nrow(vehicles)),]
split=floor(nrow(vehicles)/3)
ensembleData=vehicles[0:split,]						#Training dataset for all our models
blenderData =vehicles[(split+1):(split*2),]			#To blend all these probabilities
testingData =vehicles[(split*2+1):nrow(vehicles),]	#To see how well we did.


labelName="cylinders"
colNames=names(ensembleData)
predictors=colNames[colNames!=labelName]

### Use the "diverse" models - Train all the ensemble models - gbm,rpart and treebag
myCtrl=trainControl(method="cv"
				   ,number=3
				   ,repeats=1
				   ,returnResamp="none")

preds=ensembleData[,predictors]
target=ensembleData[,labelName]

model_gbm=train(preds,target,method="gbm",trControl=myCtrl)
model_rpart=train(preds,target,method="rpart",trControl=myCtrl)
model_treebag=train(preds,target,method="treebag",trControl=myCtrl)

### Make Predictions on the blender and testing data- added new cols having probabilities.

#So now for each case - we have probability information from 3 different models 
#Thus we have captured extra information from the other models too.
blenderData$gbmProb=predict(object=model_gbm,newdata=blenderData[,predictors])
blenderData$rpartProb=predict(object=model_rpart,newdata=blenderData[,predictors])
blenderData$treebagProb=predict(object=model_treebag,newdata=blenderData[,predictors])
testingData$gbmProb=predict(object=model_gbm,newdata=testingData[,predictors])
testingData$rpartProb=predict(object=model_rpart,newdata=testingData[,predictors])
testingData$treebagProb=predict(object=model_treebag,newdata=testingData[,predictors])

#Now we will use say "gbm" on the new blended data...for training and generating an "ensembled model"
labelName="cylinders"
colNames=names(blenderData)
predictors=colNames[colNames!=labelName]
finalBlenderModel=train(blenderData[,predictors]
					   ,blenderData[,labelName]
					   ,method="gbm"				#Better to use a model not used in the ensemble.
					   ,trControl=myCtrl)

### Make predictions using the Blended model
preds=predict(object=finalBlenderModel,newdata=testingData[,predictors])
auc=roc(testingData[,labelName],preds);auc

#Check with single models...to see the improvement...