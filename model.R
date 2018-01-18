csv.load <- function(file_name, features_name = c()){
  data = read.csv(file_name)
  return(data)
}

csv.save <- function(data){
  df = data.frame(1:length(data$label), data$label)
  colnames(df) <- c("ImageId", "Label")
  write.table(df, 'out.csv', sep = ',', row.names = FALSE)
}

# 1) Load training and testing dataset from csv files
print("Load dataset from csv files.")
data = csv.load("data/train.csv")
data_test = csv.load("data/test.csv")

# Keep just a part of training dataset
N = 42000
data = head(data, N) # Keep the N first examples

# Keep in mind the starting labels
starting_labels = data$label

# 2) Get one vs all labels for each class
print("Extract one vs all label for each class.")
labels = matrix(c(TRUE), nrow=length(data$label)[1], ncol=10)
for (i in 1:dim(labels)[2]) {
  labels[,i] = as.integer(data$label == i - 1);
}

# 3) For each class process logistic regression and compute the prediction
print("Create one model for each class.")
prediction = matrix(nrow=N, ncol=10)
prediction_test = matrix(nrow=28000, ncol=10)
for (i in 1:dim(labels)[2]) {
  print(c("    Model number : ", i - 1))

  # Train the i-th model
  data$label = labels[,i]
  result = glm(label ~ ., data, family=binomial(logit))
  prediction[,i] = result$fitted

  # Compute prediction for i-th model
  result_test = predict(result, newdata=data_test)
  prediction_test[,i] = result_test
}

print(dim(prediction))
# 4) Among all the classifiers, get the highest scoring prediction
for (i in 1:dim(labels)[1]) {
  prediction[i,1] = which.max(prediction[i,]) - 1
}

for (i in 1:dim(data_test)[1]) {
  prediction_test[i,1] = which.max(prediction_test[i,]) - 1
}

accuracy = as.integer(prediction[,1] == starting_labels)

print("Accuracy in training set : ")
print(mean(accuracy))

data_test$label = prediction_test[,1];

csv.save(data_test)
