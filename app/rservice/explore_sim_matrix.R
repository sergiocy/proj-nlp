

library(data.table)


FILE_RDS_W1_SIM_PROCESSED <- "C:/sc-sync/projects/proj-nlp/data/output/w1_sim_processed.rds"

data <- readRDS(FILE_RDS_W1_SIM_PROCESSED)


ranking <- as.data.frame(data[[1]])
scores <- as.data.frame(data[[2]])
rm(data)


head(ranking)
dim(ranking)
head(scores)
dim(scores)


####
#### first exploration
hist(scores$first)

dim(ranking[ranking$first == "1", ])
dim(ranking[ranking$second == "1", ])
dim(ranking[ranking$third == "1", ])
dim(ranking[ranking$fourth == "1", ])
dim(ranking[ranking$fifth == "1", ])

#table()

#### 35 rows with one "1"
35/nrow(ranking)

summary(ranking)



####
#### ...dataset coding cases > or < than median...
median <- sapply(1:ncol(scores), function(c){ ifelse(scores[ , c] > median(scores[ , c]), 'h', 'l') })
colnames(median) <- colnames(ranking)
rownames(median) <- rownames(ranking)



