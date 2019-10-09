

library("data.table")


FILE_RDS_W1_SIM <- "C:/sc-sync/projects/proj-nlp/data/output/w1_sim.rds"



sim <- readRDS(FILE_RDS_W1_SIM)  
head(sim)
dim(sim)

####
#### ...we get the unique rows (some words are repeated)
sim_unique <- unique(sim)[ , unique(colnames(sim)), with = FALSE]






ranking <- sapply(1:10, function(iter_rows){c(colnames(sort(subset(sim_unique[iter_rows, ], select = -c(word)), decreasing = TRUE))[1:5])})



length(sim_unique$word)