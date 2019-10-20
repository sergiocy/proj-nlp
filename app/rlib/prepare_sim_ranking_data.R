

library("data.table")


FILE_RDS_W1_SIM <- "C:/sc-sync/projects/proj-nlp/data/output/w1_sim.rds"
FILE_RDS_W1_SIM_PROCESSED <- "C:/sc-sync/projects/proj-nlp/data/output/w1_sim_processed.rds"



sim <- readRDS(FILE_RDS_W1_SIM)  
head(sim)
dim(sim)

####
#### ...we get the unique rows (some words are repeated)
sim_unique <- unique(sim)[ , unique(colnames(sim)), with = FALSE]
head(sim_unique)
dim(sim_unique)





####################################################
#### ...computing first 5 similariest words...
ranking <- sapply(1:length(sim_unique$word), function(iter_rows){
                                    podium <- colnames(sort(subset(sim_unique[iter_rows, ], select = -c(word)), decreasing = TRUE))[1:5]
                                    podium <- gsub("comp_", "", podium)
                                    #print(podium)
                                    return(podium)
                                                    })
ranking <- as.data.frame(ranking)
rownames(ranking) <- c("first", "second", "third", "fourth", "fifth")
colnames(ranking) <- sim_unique$word

ranking <- t(ranking)
head(ranking)
dim(ranking)

ranking <- cbind(ranking, rownames(ranking))
ranking <- t(apply(ranking, 1, function(r){
    #print(r)
    
    eq_or_not <- r[1:5] == r[6]
    # r <- ifelse(, 1, r)
    r2 <- ifelse(eq_or_not == TRUE, 1, r)
    return(r2)
}))

head(ranking)
dim(ranking)



####
#### ...computing scores of first 5 similariest words...
ranking_scores <- sapply(1:length(sim_unique$word), function(iter_rows){
    podium_scores <- as.vector(sort(subset(sim_unique[iter_rows, ], select = -c(word)), decreasing = TRUE)[, 1:5])
    colnames(podium_scores) <- NULL
    #print(podium)
    return(as.numeric(podium_scores))
})
ranking_scores <- as.data.frame(ranking_scores)
rownames(ranking_scores) <- c("first", "second", "third", "fourth", "fifth")
colnames(ranking_scores) <- sim_unique$word

ranking_scores <- t(ranking_scores)
head(ranking_scores)
dim(ranking_scores)





#########################################################
#### save processed data
data_sim_podium <- list(ranking = ranking, scores = ranking_scores)
rm(ranking, ranking_scores)
saveRDS(data_sim_podium, FILE_RDS_W1_SIM_PROCESSED)

















