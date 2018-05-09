setwd("/home/ajadhav/Desktop/Rossman/")
rm(list=ls())
cat("reading the train and test data (with data.table) \n")
train <- read.csv("train.csv")
test  <- read.csv("test.csv")
store <- read.csv("store.csv")

train = train[!(train$Open == "0"),] 
train <- merge(train,store,by="Store")
test <- merge(test,store,by="Store")
train2 = train[c("Date","Sales")]

cat("Data Loading . . . . . .\n")
summary(train)
cat("Pre - Processing\n")
summary(test)

train['Month'] = as.numeric(format(as.Date(train$Date), "%m"))
train['Day'] = as.numeric(format(as.Date(train$Date), "%d"))
train['Year'] = as.numeric(format(as.Date(train$Date), "%Y"))

test['Month'] = as.numeric(format(as.Date(test$Date), "%m"))
test['Day'] = as.numeric(format(as.Date(test$Date), "%d"))
test['Year'] = as.numeric(format(as.Date(test$Date), "%Y"))

train$CompetitionDistance[is.na(train$CompetitionDistance)] = mean(train$CompetitionDistance,na.rm=T)
test$CompetitionDistance[is.na(test$CompetitionDistance)] = mean(test$CompetitionDistance,na.rm=T)

train$Sales = log1p(train$Sales)
train$CompetitionDistance = log1p(train$CompetitionDistance)
test$CompetitionDistance = log1p(test$CompetitionDistance)


train = subset(train, select=-c(Date,Customers))
train$Assortment = as.numeric(train$Assortment)
train = subset(train, select=-c(PromoInterval,StateHoliday,Open))
#-----------------------------------------------------------
trainbackup = train
testbackup = test
#------------------------------------
trainA = train[(train$StoreType == "a"),] 
trainA = subset(trainA, select=-c(StoreType))

trainB = train[(train$StoreType == "b"),] 
trainB = subset(trainB, select=-c(StoreType))

trainC = train[(train$StoreType == "c"),] 
trainC = subset(trainC, select=-c(StoreType))

trainD = train[(train$StoreType == "d"),] 
trainD = subset(trainD, select=-c(StoreType))

testA = test[(test$StoreType == "a"),] 
testB = test[(test$StoreType == "b"),] 
testC = test[(test$StoreType == "c"),] 
testD = test[(test$StoreType == "d"),] 

library(xgboost)
#=========================================================
train = trainA
test = testA
feature.names <- names(train)[c(1:14)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

param <- list(  objective           = "reg:linear", 
                booster 	    = "gbtree",
                eta                 = 0.024, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.75, # 0.7
                colsample_bytree    = 0.7 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
h<-sample(nrow(train),50000)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)


clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 2000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred1 <- predict(clf, data.matrix(test[,feature.names])) 
#-----------------------------------------------------------------
#=========================================================
train = trainB
test = testB
feature.names <- names(train)[c(1,2,5:20)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

param <- list(  objective           = "reg:linear", 
                booster 	    = "gbtree",
                eta                 = 0.024, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.75, # 0.7
                colsample_bytree    = 0.7 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
h<-sample(nrow(train),50000)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)


clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 2000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred2 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
#-----------------------------------------------------------------
#=========================================================
train = trainC
test = testC
feature.names <- names(train)[c(1,2,5:20)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

param <- list(  objective           = "reg:linear", 
                booster 	    = "gbtree",
                eta                 = 0.024, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.75, # 0.7
                colsample_bytree    = 0.7 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
h<-sample(nrow(train),50000)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)


clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 2000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred3 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
#-----------------------------------------------------------------
#=========================================================
train = trainD
test = testD
feature.names <- names(train)[c(1,2,5:20)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

param <- list(  objective           = "reg:linear", 
                booster 	    = "gbtree",
                eta                 = 0.024, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.75, # 0.7
                colsample_bytree    = 0.7 # 0.7
                # alpha = 0.0001, 
                # lambda = 1
)
h<-sample(nrow(train),50000)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)


clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 2000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred4 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
#-----------------------------------------------------------------