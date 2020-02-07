

#### function to compute vector composition
#### get a dataframe with this structure (id | word | id_token | token | col_1 | col_2| ... | vec_dim_1 | ... vec_dim_n)
#### where we have several registers for each id (or word): one register for each word that compound the definition (or phrase)
#### In this way we have assoiacted each id-word to set of words builded by id_token-token


compute_vector_representation_composition <- function(dt # the 
                                                      , partition_var # string parameter (¿type list?) to define the variable to select the sets of vector to compose
                                                      , vector_col # vector with colnames with vector representation elements/dimensions
                                                      #, partition_order # parameter to define the order in composition
                                                      , type = "sum"
                                                      , file_save_rds = NA){
    
    # dt <- data_w2v_def ; partition_var <- "w"  ; vector_col <- paste0("dim_", c(1:300)) ; type <- "sum" ; save_rds = NA
    # vector_col <- c(3:302)
    
    
    #### ...get colnames from vector_col... with partition_var output dataset will be builded...
    if (is.numeric(vector_col)){
        vector_col <- colnames(dt)[vector_col]    
    }

    #### ...dataset partition values...
    partitions <- unique(dt[[partition_var]]) #### elements numeric or string 

    #### ...dataset to store results...
    df_result_composition <- data.frame()
    
    for (p in partitions){
        data_part <- dt[dt[[partition_var]] == p]
        
        if (type == "sum"){
            vec_sum <- sapply(vector_col, 
                                function(i_col){
                                    sum(data_part[[i_col]])
                                    }
                                )
            vec_sum <- cbind(p, as.data.frame(matrix(vec_sum, nrow = 1)))
            df_result_composition <- rbind(df_result_composition, vec_sum)
        } else if (type == "avg"){
            vec_avg <- sapply(vector_col, 
                              function(i_col){
                                  mean(data_part[[i_col]])
                              }
                            )
            vec_avg <- cbind(p, as.data.frame(matrix(vec_avg, nrow = 1)))
            df_result_composition <- rbind(df_result_composition, vec_avg)
        } else {
            print("arg 'type' not exist")
            stop()
        }
        #print(paste0("word ", data_w[id_w, .(id, w), ]$id, " '", data_w[id_w, .(id, w), ]$w, "' computed"))
    }
    
    
    #### ...we put colnames = partition_var + vector_col...
    colnames(df_result_composition) <- c(partition_var, vector_col)
    
    
    if (!is.na(file_save_rds)){
        tryCatch({
            print("saving dataset")
            saveRDS(df_result_composition, paste0(file_save_rds))
        },
        error = function(err){
            print("ERROR saving dataset")
        })
    } 

    #### ...remove input dataset to clean memory...
    rm(dt)
    
    
    return (df_result_composition)
}


