# Data_Analysis_R
#App Store Data Analysis Using R
library(forecast)
library(leaps)
app.df <- read.csv("AppleStore.csv")
View(app.df)


#select variables
selected.var <- c(10,14,15,17,18,19,21,22)

# partition data
set.seed(1)  # set seed for reproducing the partition
numberOfRows <- nrow(app.df)
train.index <- sample(numberOfRows, numberOfRows*0.6)  
train.df <- app.df[train.index, selected.var]
valid.df <- app.df[-train.index, selected.var]

#:regression with only transformed dataset
app.lm <- lm(log_rating_count ~ ., data = train.df)
options(scipen = 999)
summary(app.lm)

# use predict() to make predictions on a new set. 
app.lm.pred <- predict(app.lm, valid.df)
some.residuals <- valid.df$log_rating_count[1:20] - app.lm.pred[1:20]
data.frame("Predicted" = app.lm.pred[1:20], "Actual" = valid.df$log_rating_count[1:20],
           "Residual" = some.residuals)

options(scipen=999, digits = 3)
accuracy(app.lm.pred, valid.df$log_rating_count)

scatter.smooth(train.df$user_rating_ver,train.df$log_rating_count)


# model1:use step() to run stepwise regression with transformed dataset.
app.lm.step <- step(app.lm, direction = "both")
summary(app.lm.step)  # Which variables were dropped/added?
app.lm.step.pred <- predict(app.lm.step, valid.df)
accuracy(app.lm.step.pred, valid.df$log_rating_count)


#regression model that uses all transformed and untransformed variables

set.seed(1)  # set seed for reproducing the partition
selected.var2 <- c(6,9,10,14:22)
numberOfRows <- nrow(app.df)
train.index <- sample(numberOfRows, numberOfRows*0.6)  
train.df2 <- app.df[train.index, selected.var2]
valid.df2 <- app.df[-train.index, selected.var2]

app2.lm <- lm(log_rating_count ~ ., data = train.df2)
options(scipen = 999)
summary(app2.lm)

#model2:regression model that uses all transformed and untransformed variables stepwise
app2.lm.step <- step(app2.lm, direction = "both")
summary(app2.lm.step)  # Which variables were dropped/added?
app2.lm.step.pred <- predict(app2.lm.step, valid.df2)
accuracy(app2.lm.step.pred, valid.df2$log_rating_count)

#logistic
library(gains)
library(caret)
selected.var <- c(10,14,15,17,19,21,22,23)

# partition data
set.seed(1)  # set seed for reproducing the partition
numberOfRows <- nrow(app.df)
train.index <- sample(numberOfRows, numberOfRows*0.6)  
train.df <- app.df[train.index, selected.var]
valid.df <- app.df[-train.index, selected.var]
#logistic with tranformed dataset
logitI.reg <- glm(success ~ ., data = train.df, family = "binomial") 
options(scipen=999)
summary(logitI.reg)
confusionMatrix(table(predict(logitI.reg, newdata = valid.df, type="response") >= 0.5, valid.df$success == 1))
#model3:logistic with transformed dataset stepwise
logitI.reg.step <- step(logitI.reg, direction = "both")
summary(logitI.reg.step) 
confusionMatrix(table(predict(logitI.reg.step, newdata = valid.df, type="response") >= 0.5, valid.df$success == 1))


#logistic stepwise with all transformed and utransformed variables
selected.var3 <- c(6,9,10,14:17,19:23)
# partition data
set.seed(1)  # set seed for reproducing the partition
numberOfRows <- nrow(app.df)
train.index <- sample(numberOfRows, numberOfRows*0.6)  
train.df3 <- app.df[train.index, selected.var3]
valid.df3 <- app.df[-train.index, selected.var3]

logitI.reg2 <- glm(success ~ ., data = train.df3, family = "binomial") 
options(scipen=999)
summary(logitI.reg2)
confusionMatrix(table(predict(logitI.reg2, newdata = valid.df3, type="response") >= 0.5, valid.df3$success == 1))
#model4:logistic stepwise with all transformed and utransformed variables stepwise
logitI.reg2.step <- step(logitI.reg2, direction = "both")
summary(logitI.reg2.step) 
confusionMatrix(table(predict(logitI.reg2.step, newdata = valid.df3, type="response") >= 0.5, valid.df3$success == 1))
