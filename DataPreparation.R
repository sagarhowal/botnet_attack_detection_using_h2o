

#Load files

benign <- read.csv("benign_traffic.csv")
benign$class <- rep(0)

junk <- read.csv("gafgyt_attacks/junk.csv")
junk$class <- rep(1)

scan <- read.csv("gafgyt_attacks/scan.csv")
scan$class <- rep(1)

# tcp <- read.csv("gafgyt_attacks/tcp.csv")
# tcp$class <- rep(1)
# 
# udp <- read.csv("gafgyt_attacks/udp.csv")
# udp$class <- rep(1)


data <- rbind(benign, junk, scan) #Add TCP and UDP if required

str(data)

#convert class label to factor
data$class <- as.factor(data$class)
class(data$class) #Factor. Moving ahead
table(data$class)

#Write Data
write.csv(data, "data.csv")


#Sample 40% of the dataset for modelling
library(caret)
set.seed(42)
sampleIndex <- createDataPartition(data$class, p = .4, 
                                  list = FALSE, 
                                  times = 1)
head(sampleIndex)

data_small <- data[ sampleIndex,]

write.csv(data_small, "data_small.csv")


#Create Train and Test for the SMALL DATASET

trainIndex <- createDataPartition(data_small$class, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

train <- data_small[ trainIndex,]
test  <- data_small[-trainIndex,]

write.csv(train, "train.csv")
write.csv(test, "test.csv")



















