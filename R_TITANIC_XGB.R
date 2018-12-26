
library(tidyverse)
library(xgboost)

# read data ====================================================

data.test <- read_csv('data/test.csv')
data.train <- read_csv('data/train.csv')

# find missing values ==========================================
data.missing <- (
    data.test %>%
    summarise_all(funs(sum(is.na(.)))) %>%
    gather() %>%
    rename(VariableName = key, Count=value) %>%
    mutate(Info = 'Missing', Source='test')
  ) %>%
  bind_rows(
    data.train %>%
      summarise_all(funs(sum(is.na(.)))) %>%
      gather() %>%
      rename(VariableName = key, Count=value) %>%
      mutate(Info = 'Missing', Source='train')
  )

ggplot(data = data.missing, 
       aes(x = VariableName, y = Count, fill = Source)) +
  geom_bar(stat="identity") +
  geom_text(aes(label = Count), vjust=0) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# wrangling ====================================================

# tag origin and merge datasets so we can clean the data
data.full <- (data.train %>% mutate(Source = 'train')) %>%
  bind_rows((data.test) %>% mutate(Source = 'test'))

# Age is the main variable we need to impute
# assume there may be variations between Class, Where they embarked, and we
# can use the Title of their name. Find the median Ages of each group and
# use that to fill in NAs 

# separate the surname, title and forenames
data.model <- data.full %>%
  # 2 records have no Embarkation, looking by eye assign them to Southampton
  mutate(Embarked = if_else(PassengerId %in% c(62,830), 'S', Embarked)) %>%
  # separate the Surname delimited by ,
  separate(Name, c("Surname", "RestName"), ",") %>%
  # trim whitespace
  mutate(RestName = trimws(RestName)) %>%
  # separate the Title from the rest of the name
  separate(RestName, c("Title", "ForeNames"), "\\.", extra = "merge") %>%
  # record if Age was Imputed or not
  mutate(ImputeAge = if_else(is.na(Age),0,1)) %>%
  # there's one Ms, assume thats a Miss
  mutate(Title = if_else(Title == 'Ms', 'Miss', Title)) %>%
  # Make a group 
  mutate(grp = paste(Embarked, Pclass, Title))

# calculate median of age within groups 
data.ages <- data.model %>%
  filter(!is.na(Age)) %>%
  group_by(grp) %>%
  summarise(n(), min(Age), max(Age), mean(Age), med = median(Age))

# fill in missing ages from nearest neighbour median
data.temp <- data.model %>%
  left_join(data.ages) %>%
  mutate(Age = if_else(is.na(Age), med, Age)) 

# best data so far
data.model <- data.temp %>%
  select(Source, Survived, 
         PassengerId, Title, ForeNames, Surname,
         Sex, Age, SibSp, Parch,
         Pclass, Embarked, ImputeAge)

rm(data.temp, data.ages)
# eda ==========================================================

data.plot.input <- data.model %>%
  filter(Source == 'train')

# get crosstab figures
data.plot <- data.plot.input %>%
  group_by(Pclass) %>%
  count(Survived) %>%
  mutate(pcent = round(n / sum(n) * 100, 1)) %>%
  mutate(Label = paste0(pcent,'%')) %>%
  mutate(Survived = as.character(Survived)) %>%
  select(Pclass, Survived, pcent, Label)
  
# column totals
data.tot <- count(data.plot.input, Pclass) %>%
  mutate(Survived='Total', pcent = n) %>%
  mutate(Label = as.character(pcent)) %>%
  select(Pclass, Survived, pcent, Label)

# merge
data.plot <- data.plot %>%
  bind_rows(.,data.tot)

data.plot$Survived <- factor(data.plot$Survived, levels = c('0','1','Total'))

# plot
ggplot(data.plot, aes(x = Pclass, y = Survived, fill = pcent)) +
  # tile with black contour
  geom_tile(color="black") + 
  # B&W theme, no grey background
  theme_bw() + 
  # square tiles
  coord_equal() + 
  # Green color theme for `fill`
  scale_fill_distiller(palette="Greens", direction=1) + 
  # printing values in black
  geom_text(aes(label = Label), color="black") +
  # removing legend for `fill` since we're already printing values
  guides(fill=F) +
  # since there is no legend, adding a title
  labs(title = "Survived by pclass")

rm(data.plot, data.plot.input)

# xgboost ======================================================

# remove Fare as it doesn't seem logical from the values, 
# and Cabin, Ticket, PassengerId add nothing

# get data in the correct format
# data must be a sparse matrix with no nas
data.train.model <- data.model %>%
  filter(Source == 'train') %>%
  select(-Source, -PassengerId, -Title, -ForeNames, -Surname)

dim(data.train.model)
sparse_x <- Matrix::sparse.model.matrix(Survived~.-1, 
                                        data = data.train.model)

dim(sparse_x)
data.test.model <- data.model %>%
  filter(Source == 'test') %>%
  select(-Source, -Survived, -PassengerId, -Title, -ForeNames, -Surname)

sparse_y <- Matrix::sparse.model.matrix(~.-1, 
                                        data = data.test.model)

# targets must be numeric
targets_y <- data.train %>%
  select(Survived) %>%
  pull()

# simple model ------------------------------------------
bstSparse1 <- xgboost(data = sparse_x, 
                     label = targets_y, 
                     max.depth = 2, 
                     eta = 1, 
                     nthread = 2, 
                     nrounds = 2, 
                     objective = "binary:logistic")

importance.xgb.1 <- xgb.importance(feature_names = sparse_x@Dimnames[[2]], 
                             model = bstSparse1)

xgb.plot.importance(importance_matrix = importance.xgb.1)
xgb.plot.tree(model = bstSparse1)

# predict
pred1 <- predict(bstSparse1, sparse_y)


#another set of hyperparameters
bstSparse2 <- xgboost(data = sparse_x, label = targets_y, 
                      max.depth = 15, eta = 1, nthread = 2, 
                      nround = 30, objective = "binary:logistic",
                      min_child_weight = 50)
importance.xgb.2 <- xgb.importance(feature_names = sparse_x@Dimnames[[2]], 
                                   model = bstSparse2)

xgb.plot.importance(importance_matrix = importance.xgb.2)
xgb.plot.tree(model = bstSparse2)
# predict
pred2 <- predict(bstSparse2, sparse_y)

# and another
#another set of hyperparameters
bstSparse3 <- xgboost(data = sparse_x, label = targets_y, 
                      max.depth = 20, eta = 1, nthread = 2, 
                      nround = 50, objective = "binary:logistic",
                      min_child_weight = 50)
importance.xgb.3 <- xgb.importance(feature_names = sparse_x@Dimnames[[2]], 
                                   model = bstSparse3)

# predict
pred3 <- predict(bstSparse3, sparse_y)

# compare
pred <- data.frame(data.test$PassengerId, pred1, pred2, pred3) %>%
  mutate(pred1B = if_else(pred1 < 0.5, 0, 1),
         pred2B = if_else(pred2 < 0.5, 0, 1),
         pred3B = if_else(pred3 < 0.5, 0, 1)) 

ggplot(pred, aes(pred3, pred2, colour=pred1B)) +
  geom_point()


