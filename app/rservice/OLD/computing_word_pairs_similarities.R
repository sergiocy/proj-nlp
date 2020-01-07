

library("reticulate")
library("data.table")




FILE_PY_READ_PICKLE <- "C:/sc-sync/projects/proj-nlp/app/service/util/read_pickle.py"
FILE_PXL_TO_READ <- "C:/sc-sync/projects/proj-nlp/data/output/combined-definitions-complete"
FILE_W1_SIM <- "C:/sc-sync/projects/proj-nlp/data/output/w1_sim.rds"

#### read dataframe from .pkl file
source_python(FILE_PY_READ_PICKLE)
pickle_data <- read_pickle_file(FILE_PXL_TO_READ)

data <- setDT(pickle_data)
rm(pickle_data)

#dim(data)
#str(data)
#colnames(data)



#### TODO: add R-packages in environment conda
#### TODO: call this script from python /argument parsing
#### TODO: add log-file




compute_similarity_cosin_between_vectors <- function(v1, v2, str1 = 'word1', str2 = 'def'){
    
    sim <- tryCatch(
        {
            # num <- v1*v2, but we test sapply function
            num <- sum(sapply(1:length(v2), function(i) { v1[i]*v2[i] })) 
            den1 <- sqrt( sum(sapply(1:length(v2), function(i) { v1[i]*v1[i] })) )
            den2 <- sqrt( sum(sapply(1:length(v2), function(i) { v2[i]*v2[i] })) )
            
            return (num/(den1*den2))
        },
        error=function(e) {
            message(paste0("ERROR"))
            # Choose a return value in case of error
            return(NA)
        },
        #warning=function(cond) {
        #    
        #},
        finally={
            #message(paste0("Computed similariry between '", str1, "' - '", str2))
        }
    )
    
    return(round(as.numeric(sim), 4))
}

# compute_similarity_cosin_between_vectors(c(1,0), c(0,1))



####
#### ...we select word1 and definitions (from composition of definition words) for it
data_sim_w1 <- subset(data, select = c(w1, w1_vectorized, def1_vector_sum
                                       #, def1_vector_avg
                                       ))
#length(data_sim_w1[1, , ]$def1_vector_sum)
colnames(data_sim_w1) <- c("w", "vec_w", "vec_composed")


####
#### compute similarities matrix
matrix_sim <- do.call("rbind", sapply(1:length(data_sim_w1$w)
                                             , function(iter_on_words) {
                                                 word = data_sim_w1$w[iter_on_words]
                                                 # vector associated to one only word
                                                 vec_word <- unlist(data_sim_w1[ iter_on_words, .(vec_w), ][[1]]) 
                                                 
                                                 # ...we compute similarity between a word an a set of composition-vectors
                                                 vec_word_vs_composed <- sapply(1:length(data_sim_w1$vec_composed)
                                                                                , function(i_def){
                                                                                    vec_def <- unlist(data_sim_w1[ i_def, .(vec_composed), ][["vec_composed"]])
                                                                                    return (compute_similarity_cosin_between_vectors(vec_word, vec_def))
                                                                                })
                                                 
                                                 print(paste0("definition ", iter_on_words, " computed"))
                                                 return(list(c(word, vec_word_vs_composed)))
                                             })
                            )
matrix_sim <- as.data.table(matrix_sim)
colnames(matrix_sim) <- c("word", sapply(1:length(data_sim_w1$w), function(i_def){paste0("comp_", data_sim_w1$w[i_def])}))

#fwrite(matrix_sim, file = "")
saveRDS(matrix_sim, FILE_W1_SIM)    


    




