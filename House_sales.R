library(caret)
library(stats)
install.packages("lubridate")
library(lubridate)
install.packages("data.table")
library(data.table)
library(ggplot2)
install.packages("ggcorrplot")
library(ggcorrplot)


setwd("C:/Users/sbasapurashiva8127/Documents")

#read in csv to data frame called 'house'
house<- read.csv("kc_house_data.csv")

#review data structure
str(house)
names(house)
head(house)
summary(house)

#training the dataset


house$date<-format(as.Date(house$date, format = "%Y%m%d"), "%m/%d/%Y")

set.seed(123)
trainIndex <- createDataPartition(house$price, p = .8, list = FALSE, times = 1)

train_data <- house[ trainIndex,]
test_data  <- house[-trainIndex,]
str(train_data)
str(test_data)


summary(train_data)
summary(test_data)

head(train_data)

# to check missing records
sapply(train_data, function(x) sum(is.na(x)))

# to check any duplicate records
any(duplicated(train_data))

# Identify integer columns in train data.frame
int_cols <- sapply(train_data, is.integer)

# Convert integer columns to numeric
train_data[int_cols] <- lapply(train_data[int_cols], as.numeric)

# Identify integer columns in test data.frame
int_cols <- sapply(test_data, is.integer)

# Convert integer columns to numeric
test_data[int_cols] <- lapply(test_data[int_cols], as.numeric)

train_data$date<-mdy(train_data$date)
train_data$year<-year(train_data$date)
train_data$month<-month(train_data$date)
train_data$days_since<-as.numeric(Sys.Date() - train_data$date)

test_data$date<-mdy(test_data$date)
test_data$year<-year(test_data$date)
test_data$month<-month(test_data$date)
test_data$days_since<-as.numeric(Sys.Date() - test_data$date)

summary(train_data)

# Example for building a linear regression model
model <- lm(price ~ bedrooms + bathrooms + sqft_living  + waterfront + grade + yr_built + view  + lat + long + zipcode, data = train_data)

plot(house$price ~ house$sqft_living )

summary(model)

# Example for model evaluation
predictions <- predict(model, test_data)
summary(predictions)


# Calculate evaluation metrics
mse <- mean((predictions - test_data$price)^2)
rmse <- sqrt(mse)
mae <- mean(abs(test_data$price - predictions))
r_squared <- 1 - (sum((test_data$price - predictions)^2) / sum((test_data$price - mean(test_data$price))^2))

summary(mse)
summary(rmse)
summary(mae)
summary(r_squared)



###############################

# Ridge regression

# Load the necessary packages
library(glmnet)

# Load the data and split into training and testing sets
house <- read.csv("kc_house_data.csv")
train_index <- sample(nrow(house), round(nrow(house)*0.8))
train_data <- house[train_index, ]
test_data <- house[-train_index, ]



# Separate target variable and input features for training dataset
x_train <- as.matrix(train_data[, c("bedrooms", "bathrooms", "sqft_living", "grade", "yr_built", "view", "lat", "long", "zipcode")])
y_train <- as.matrix(train_data$price)

# Fit the Ridge regression model with cross-validation to tune the regularization parameter alpha
cv_fit <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)
best_alpha <- cv_fit$lambda.min
ridge <- glmnet(x_train, y_train, alpha = 0, lambda = best_alpha)

# Separate target variable and input features for testing dataset
x_test <- as.matrix(test_data[, c("bedrooms" , "bathrooms", "sqft_living", "grade", "yr_built" ,"view", "lat", "long" , "zipcode")])
y_test <- as.matrix(test_data$price)

# Predict the values on the testing set using the Ridge regression model with the optimal alpha
predictions <- predict(ridge, newx = x_test)

# Calculate the accuracy metrics
rmse_rr <- sqrt(mean((y_test - predictions)^2))
cat("RMSE of Ridge Regression: ", rmse_rr, "\n")

r2 <- cor(y_test, predictions)^2
mae <- mean(abs(y_test - predictions))
cat("R-squared of Ridge Regression: ", r2, "\n")
cat("MAE of Ridge Regression: ", mae, "\n")



