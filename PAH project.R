set.seed(8)

#importing libraries

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, caret, data.table, dplyr, yardstick, 
               e1071, pROC)

#importing the dataset

genes <- read.csv("~/Documents/MSc Data Analytics/Data Mining and ML(I)/divers/project/GDS504.clean.csv")

#selecting subset for analysis

genes <- genes[601:1600,]

#dropping the id columns

genes[1:2] <- NULL

#transposing the dataset

genes <- as.data.frame(t(genes))

#creating a column of labels
#first 14 rows == PAH == 1
#last 6 rows == no PAH == 0

genes$labels <- NA

genes[1:14, 1001] <- 1
genes[15:20, 1001] <- 0

#converting the target into a factor

genes[,1001] <- as.factor(genes[,1001])

#fixing illegal variable names issues

names(genes) <- make.names(names(genes))

#checking for NAs

anyNA(genes)

#splitting the dataset

trainIndex <- createDataPartition(genes$labels, p = 0.5, 
                                  list = FALSE)

train.set <- genes[ trainIndex,]
test.set  <- genes[-trainIndex,]

dim(train.set)
dim(test.set)

#plotting the distribution of the response variable

ggplot(data=train.set, aes(x=labels, col=labels))+ 
  geom_bar(stat="count", fill='white')+
  labs(title='Labels Distribution - PAH Dataset', x='labels')

#multidimensional scaling for visualisation
#creating a distance matrix

genes_square <- dist(train.set[,-1001])

#computing the scaling

genes_mds <- cmdscale(genes_square, list. = T)

#plotting the dataset

PAH <- train.set[,1001]

ggplot(as.data.frame(genes_mds$points), aes(genes_mds$points[,1], 
                                genes_mds$points[,2], colour=PAH))+
  geom_point()+
  labs(title='Detection of PAH in 10 Individuals - Multidimensional Scaling',
        x ='Scaled x Coordinates', y = 'Scaled y Coordinates')

#analysing multicollinearity

highly.correlated <- findCorrelation(cor(train.set[,1:1000]), cutoff=0.75)
                                     
length(highly.correlated)

#filtering features with the RF algorithm

rfe.pah <- rfe(train.set[,1:1000], train.set[,1001], sizes=10:50, 
               rfeControl=rfeControl(functions=rfFuncs, method="LOOCV"))

predictors(rfe.pah)

length(predictors(rfe.pah))

#filtering/updating the training & test set

train.set <- select(train.set, c(predictors(rfe.pah), labels))
test.set <- select(test.set, c(predictors(rfe.pah), labels))

#fitting the knn model

knn.model <- train(labels ~., data = train.set, method = "knn",
                   trControl=trainControl('LOOCV'),
                   tuneGrid = expand.grid(k = c(3,5)))
knn.model

#calculating roc analysis on each parameter 

knn_features <- setorder(varImp(knn.model)$importance, order=-X0)
knn_features$X1 <- NULL
colnames(knn_features) <- 'ROC'
knn_features <- t(knn_features)
knn_features

#computing predictions & confusion matrix

pred.knn <- predict(knn.model, test.set[,-35])
confusionMatrix(pred.knn, test.set$labels, positive='1')

#plotting the confusion matrix

truth_predicted <- data.frame(obs=test.set$labels, pred=pred.knn)

cm <- conf_mat(truth_predicted, obs, pred)

autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low = "pink", high = "cyan")+
  ggtitle('Confusion Matrix - kNN Model')

#computing the ROC curve

roc(test.set$labels, as.numeric(as.vector(pred.knn)))

plot.roc(test.set$labels, as.numeric(as.vector(pred.knn)),
         main='ROC Curve - kNN Model')

#fitting a random forest model

rf.model <- train(labels~.,data=train.set, method = "rf",
                  trControl = trainControl(method='LOOCV', search = 'grid'), 
                  importance=T)

rf.model

#computing feature importance

varImp(rf.model)

#computing predictions & confusion matrix

pred.rf <- predict(rf.model, test.set[,-35])
confusionMatrix(pred.rf, test.set$labels, positive='1')

#plotting the confusion matrix

truth_predicted <- data.frame(obs=test.set$labels, pred=pred.rf)

cm <- conf_mat(truth_predicted, obs, pred)

autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low = "pink", high = "cyan")+
  ggtitle('Confusion Matrix - RF')

#computing ROC curve

roc(test.set$labels, as.numeric(as.vector(pred.rf)))

plot.roc(test.set$labels, as.numeric(as.vector(pred.rf)),
         main='ROC Curve - RF')

#fitting a SVM model

svm.model <- svm(labels~.,data=train.set, type="C-classification")

svm.model

#computing predictions & confusion matrix

pred.svm <- predict(svm.model, test.set[,-35])
confusionMatrix(pred.svm, test.set$labels, positive='1')


#plotting the confusion matrix

truth_predicted <- data.frame(obs=test.set$labels, pred=pred.svm)

cm <- conf_mat(truth_predicted, obs, pred)

autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low = "pink", high = "cyan")+
  ggtitle('Confusion Matrix - SVM Model')

#computing the ROC curve 

roc(test.set$labels, as.numeric(as.vector(pred.svm)))

plot.roc(test.set$labels, as.numeric(as.vector(pred.svm)), 
         main='ROC Curve - SVM Model')
