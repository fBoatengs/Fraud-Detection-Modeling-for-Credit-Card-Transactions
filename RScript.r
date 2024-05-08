
# Loading the dataset and removing the id column
fraud <- read.csv("creditcard_2023.csv")
FRAUD <- fraud[,-1]
head(FRAUD)


# Checking for missing values in the data set
colSums(is.na(FRAUD))


# Summary Statistics
print(summary(FRAUD))
  

# Creating distribution plots for the variables
options(repr.plot.width=12, repr.plot.height=10)
FRAUD %>%
  select(-Class) %>%
  gather() %>% 
  ggplot(aes(value, fill = key)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~key, scales = "free") +
  labs(title = "Distribution of Variables") +
  guides(fill = "none")
 

# Creating the correlation  plot to check the correlation  between the variables 
corr_matrix <- FRAUD %>% 
  select(-Class) %>% 
  cor()
options(repr.plot.width=12, repr.plot.height=8)


# Creating a violin plot distribution by class
options(repr.plot.width=12, repr.plot.height=20)
FRAUD %>% gather(key = "Variable", value = "Value", -Class) %>%
  mutate(Class = as.factor(Class)) %>%
  ggplot(aes(x = Class, y = Value, fill = Class)) +
  geom_violin() +
  facet_wrap(~ Variable, scales = "free") +
  labs(title = "Distribution of Variables by Class",
       x = "Class",
       y = "Value") +
  scale_fill_manual(values = c("0" = "blue", "1" = "orange")) +
  theme_minimal() +
  guides(fill = "none")


# Plotting the class distribution
options(repr.plot.width=10, repr.plot.height=7)
ggplot(FRAUD, aes(x = factor(Class), fill = factor(Class))) +
  geom_bar() +
  labs(x = "Class", y = "Count", fill = "Class", title = "Distribution of Classes") +
  scale_fill_manual(values = c("0" = "blue", "1" = "orange"), labels = c("Genuine",    "Fraudulent")) +
  scale_y_continuous(labels = scales::comma) + 

  
# Normalizing the 'Amount' varaible
preProcessRange <- preProcess(FRAUD["Amount"], method = c("center", "scale"))
FRAUD_norm <- predict(preProcessRange, FRAUD["Amount"])
FRAUD <- bind_cols(FRAUD[, -which(names(FRAUD) %in% "Amount")], FRAUD_norm)
head(FRAUD)


# Splitting the data into training and test sets (70% training, 30% test)
set.seed(123)
split <- sample.split(FRAUD$Class, SplitRatio = 0.7)
training_set <- subset(FRAUD, split == TRUE)
test_set <- subset(FRAUD, split == FALSE)

nrow(training_set)
nrow(test_set)

# Logistic regression model
set.seed(123)
glm_model = glm(Class~. ,data = training_set, family = binomial(link = 'logit'))
summary(glm_model)
pred_glm <- predict(glm_model, newdata = test_set, type = 'response')

# ROC and PR Curves for GLM  model
glm_fg <- pred_glm[test_set$Class == 1]
glm_bg <- pred_glm[test_set$Class == 0]

glm_roc <- roc.curve(scores.class0 = glm_fg , scores.class1 = glm_bg, curve = T)
glm_pr <- pr.curve(scores.class0 = glm_fg , scores.class1 = glm_bg, curve = T)

options(repr.plot.width=12, repr.plot.height=7)
par(mfrow = c(1, 2))
plot(glm_roc, col = "blue", main = "ROC Curve")
plot(glm_pr, col = "blue", main = "PR Curve")


### 2. Decision Tree
set.seed(123)
dt_model <- rpart(Class ~ ., data = training_set, method = "class")
summary(dt_model)
pred_dt <- predict(dt_model, newdata = test_set, type = "prob")


# ROC and PR Curves for decision tree  model
dt_fg <- pred_dt[test_set$Class == 1]
dt_bg <- pred_dt[test_set$Class == 0]

dt_roc <- roc.curve(scores.class0 = dt_fg , scores.class1 = dt_bg, curve = T)
dt_pr <- pr.curve(scores.class0 = dt_fg , scores.class1 = dt_bg, curve = T)

par(mfrow = c(1, 2))
plot(dt_roc, col = "blue", main = "ROC Curve")
plot(dt_pr, col = "blue", main = "PR Curve")


### 3. Random Forest
set.seed(123)
rf_model <- randomForest(Class ~ ., data = training_set, ntree = 10)
summary(rf_model)
pred_rf <- predict(rf_model, newdata = test_set, type = "class")

# ROC and PR Curves for random forest model
rf_fg <- pred_rf[test_set$Class == 1]
rf_bg <- pred_rf[test_set$Class == 0]

rf_roc <- roc.curve(scores.class0 = rf_fg , scores.class1 = rf_bg, curve = T)
rf_pr <- pr.curve(scores.class0 = rf_fg , scores.class1 = rf_bg, curve = T)

par(mfrow = c(1, 2))
plot(rf_roc, col = "blue", main = "ROC Curve")
plot(rf_pr, col = "blue", main = "PR Curve")


### 4. Neural Network
set.seed(123)
nn_model <- nnet(Class ~ ., data = training_set, size = 10, linout = FALSE, maxit = 200)
summary(nn_model)
nn_predictions_probs <- predict(nn_model, newdata=test_set[, !names(test_set) %in% "Class"], type="raw")
pred_nn <- ifelse(nn_predictions_probs > 0.5, 1, 0)

# ROC and PR Curves for random forest model
nn_fg <- pred_nn[test_set$Class == 1]
nn_bg <- pred_nn[test_set$Class == 0]

nn_roc <- roc.curve(scores.class0 = nn_fg , scores.class1 = nn_bg, curve = T)
nn_pr <- pr.curve(scores.class0 = nn_fg , scores.class1 = nn_bg, curve = T)

par(mfrow = c(1, 2))
plot(nn_roc, col = "blue", main = "ROC Curve")
plot(nn_pr, col = "blue", main = "PR Curve")


## Model Evaluation and Comparison
par(mfrow = c(1, 2))

# Extract ROC data for all models
roc_data_glm <- data.frame(FPR = glm_roc$curve[, 1], TPR = glm_roc$curve[, 2])
roc_data_dt <- data.frame(FPR = dt_roc$curve[, 1], TPR = dt_roc$curve[, 2])
roc_data_rf <- data.frame(FPR = rf_roc$curve[, 1], TPR = rf_roc$curve[, 2])
roc_data_nn <- data.frame(FPR = nn_roc$curve[, 1], TPR = nn_roc$curve[, 2])

# Plot ROC curves
plot(roc_data_glm$FPR, roc_data_glm$TPR, type = 'l', col = "orange", lwd = 5, xlab = "False Positive Rate", ylab = "True Positive Rate", main = "ROC Curves Comparison", xlim = c(0, 1), ylim = c(0, 1))
lines(roc_data_dt$FPR, roc_data_dt$TPR, col = "blue", lwd = 5)
lines(roc_data_rf$FPR, roc_data_rf$TPR, col = "brown", lwd = 5)
lines(roc_data_nn$FPR, roc_data_nn$TPR, col = "purple", lwd = 5)
legend("bottomright", legend = c("GLM", "Decision Tree", "Random Forest", "Neural Network"), col = c("orange", "blue", "brown","purple"), lty = 1, lwd = 5)

# Extract PR data for all models
pr_data_glm <- data.frame(Recall = glm_pr$curve[, 1], Precision = glm_pr$curve[, 2])
pr_data_dt <- data.frame(Recall = dt_pr$curve[, 1], Precision = dt_pr$curve[, 2])
pr_data_rf <- data.frame(Recall = rf_pr$curve[, 1], Precision = rf_pr$curve[, 2])
pr_data_nn <- data.frame(Recall = nn_pr$curve[, 1], Precision = nn_pr$curve[, 2])

# Plot PR curves
plot(pr_data_glm$Recall, pr_data_glm$Precision, type = 'l', col = "orange", , lwd = 5, xlab = "Recall", ylab = "Precision", main = "PR Curves Comparison", xlim = c(0, 1), ylim = c(0, 1))
lines(pr_data_dt$Recall, pr_data_dt$Precision, col = "blue", lwd = 5)
lines(pr_data_rf$Recall, pr_data_rf$Precision, col = "brown", lwd = 5)
lines(pr_data_nn$Recall, pr_data_nn$Precision, col = "purple", lwd = 5)
legend("bottomright", legend = c("GLM", "Decision Tree", "Random Forest", "Nueral Network"), col = c("orange", "blue", "brown", "purple"), lwd = 5)

# Reset the plotting parameters to default
par(mfrow = c(1, 1))

# Extre AUC ROC values
auc_roc_glm <- glm_roc$auc
auc_roc_dt <- dt_roc$auc
auc_roc_rf <- rf_roc$auc
auc_roc_nn <- nn_roc$auc

# Extre AUC PR values
auc_pr_glm <- glm_pr$auc.integral
auc_pr_dt <- dt_pr$auc.integral
auc_pr_rf <- rf_pr$auc.integral
auc_pr_nn <- nn_pr$auc.integral

# Create a data frame
auc_table <- data.frame(
  Model = c("GLM", "Decision Tree", "Random Forest", "Neural Network"),
  AUC_ROC = c(auc_roc_glm, auc_roc_dt, auc_roc_rf, auc_roc_nn),
  AUC_PR = c(auc_pr_glm, auc_pr_dt, auc_pr_rf, auc_pr_nn)
)

# Print the table
print(auc_table)

# Create the bar plot for the results 
par(mfrow = c(1, 2))

# Create bar plot for AUROC
barplot(auc_table$AUC_ROC, names.arg = auc_table$Model, col = c("blue", "orange", "brown", "purple"), main = "AUC_ROC", ylab = "AUROC Value", las = 2, ylim = c(0, 1))

# Create bar plot for AUPRC
barplot(auc_table$AUC_PR, names.arg = auc_table$Model, col = c("blue", "orange", "brown", "purple"), main = "AUC_PR", ylab = "AUPRC Value", las = 2, ylim = c(0, 1))

# Reset graphical parameters to default
par(mfrow = c(1, 1))
