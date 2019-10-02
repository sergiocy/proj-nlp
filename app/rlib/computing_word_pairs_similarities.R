

library("reticulate")
library("data.table")




FILE_PY_READ_PICKLE <- "C:/sc-sync/projects/proj-nlp/app/service/util/read_pickle.py"
FILE_PXL_TO_READ <- "C:/sc-sync/projects/proj-nlp/data/output/combined-definitions-complete"




#### read dataframe from .pkl file
source_python(FILE_PY_READ_PICKLE)
pickle_data <- read_pickle_file(FILE_PXL_TO_READ)


data <- setDT(pickle_data)
rm(pickle_data)

dim(data)
str(data)
colnames(data)



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
            message(paste0("Computed similariry between '", str1, "' - '", str2))
        }
    )
    
    return(sim)
}

# compute_similarity_cosin_between_vectors(c(1,2,3,5), c(1,2,3,4))




data_sim_w1 <- subset(data, select = c(w1, w2, def1_vector_sum, def1_vector_avg, def2_vector_sum, def2_vector_avg))

#### data to dev
iter_on_words <- 1
iter_definitions <-1 #### definition selection between 1 - 345

w_vec <- data_sim_w1[ 1, w1_vectorized, ][[1]][[iter_on_words]] # vector associated to one only word
    w_def <- data_sim_w1[ , def1_vectorized, ][[iter_definitions]] # set of vectors associated to each word in definition
    


#w_def <- sapply( , data_sim_w1[ , def1_vectorized, ][[iter_definitions]][[iter_words_of_one_definition]]

                 
print(w_vec)
class(w_vec)
length(w_vec[[1]][[1]])
length(w_def)


