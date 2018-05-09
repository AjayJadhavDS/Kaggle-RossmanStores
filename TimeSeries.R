
library(dplyr)

library(fpp)
train <- read.csv("train.csv", header=T, sep=",", na.strings="", stringsAsFactors=F)

test <- read.csv("test.csv", header=T, sep=",", na.strings="", stringsAsFactors=F)

store <- read.csv("store.csv", header=T, sep=",", na.strings="", stringsAsFactors=F)

train$Date <- as.Date(train$Date)

test$Date <- as.Date(test$Date)

storeList <- sort(unique(test$Store))   # 856 stores

training <- subset(train,  Sales != 0 & Store %in% storeList)   # 606707

testing <-  subset(train,  Sales != 0 & Store %in% storeList)    # 35235

training <- cbind(training, Store, Date)

testing <- arrange(testing, Store, Date)


remove(valLogResults)

valLogResults <- data.frame(Store=numeric(0),Date=as.Date(character(0)), 
                            Sales=numeric(0),forecastValues=numeric(0))

for (str in storeList) {
  
  trainingNew <- filter(training, Store == str)
  
  trainingNew <- arrange(trainingNew, Date)
  
  trainingTS <- ts(log1p(trainingNew$Sales))
  
  trainingData <- cbind(as.numeric(trainingNew$Promo),
                        as.numeric(trainingNew$DayOfWeek))
  
  colnames(trainingData) <- c("Promo",
                              "DayOfWeek")
  
  
  fit.arima <- auto.arima(trainingTS, stepwise=FALSE,approx=FALSE, xreg=trainingData)
  
  testingNew <- filter(testing, Store == str)
  
  testingNew <- arrange(testingNew, Date)
  
  testingData <- cbind(as.numeric(testingNew$Promo),
                       as.numeric(testingNew$DayOfWeek))
  
  colnames(testingData) <- c("Promo",
                             "DayOfWeek")
  
  predict.arima <- forecast(fit.arima,h=nrow(testingNew),xreg=data.frame(testingData))
  
  forecastValues <- round(expm1(predict.arima$mean[1:nrow(testingNew)]))
  
  valLogResults <- rbind(valLogResults, 
                         data.frame(Store=testingNew$Store, Date=testingNew$Date, 
                                    Sales=testingNew$Sales, forecastValues=forecastValues))
  
} 

#valResults2 <- arrange(valResults, Store, Date)


#write.csv(valResults2, "valResults.csv",row.names=F)

rmspe.val <- sqrt(sum(
  ((valLogResults$Sales - valLogResults$forecastValues)/
     (valLogResults$Sales)
  ) ^ 2
)/
  nrow(valLogResults)
)


rmspe.val