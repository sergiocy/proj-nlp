

# vector_cosines <- sapply(1:nrow(data_sim), function(r){
#     word1 <- as.character(data_sim[r, .(w1), ])
#     word2 <- as.character(data_sim[r, .(w2), ])
#     
#     index_word1 <- which(df_vec$w == word1)
#     index_word2 <- which(df_vec$w == word2)
#     
#     rep1 <- get_vector_of_w(row = index_word1)
#     rep2 <- get_vector_of_w(row = index_word2)
#     
#     cosine <- compute_cosin(rep1, rep2)
#     print(paste0("computing words - ", word1, " and ", word2, " - ", cosine))
#     
#     return (cosine)
# })



get_vector_from_df_row <- function(df_vec, df_cols, row = 1){
    setDT(df_vec)
    
    if (is.numeric(df_cols)) {
        df_cols <- colnames(df_vec[ , df_cols, with = FALSE])
    } 
    
    vector <- as.numeric(df_vec[row, df_cols, with = FALSE])
    
    return (vector)
}


compute_cosin <- function(a, b){
    cos_theta <- ( sum(a*b) / ( sqrt(sum(a * a)) * sqrt(sum(b * b)) ) )
    #theta <- acos( sum(a*b) / ( sqrt(sum(a * a)) * sqrt(sum(b * b)) ) )
    return (cos_theta)
}




#### function to compute similarities matrix from two datasets with vector represetations in its column
#### In output matrix we have df1 by rows and df2 by columns

compute_similarities_matrix <- function (
                            df1
                            , df2
                            , vector_col1
                            , vector_col2
                            , var_key1  # variable to identify rows in df1 (by rows)
                            , var_key2  # variable to identify rows in df2 (by columns)
                            , file_save_rds = NA) {
    
    # df1 <- df_w ; df2 <- df_comp ; vector_col1 <- colnames(df_w)[3:302] ; vector_col2 <- colnames(df_comp)[3:302] ; var_key1 <- "w" ; var_key2 <- "w" ;  
    
    setDT(df1)
    setDT(df2)
    
    #### ...we verify that vector have the same dimension...
    if (length(vector_col1) == length(vector_col2)){
        
        similarities <- lapply(1:nrow(df1), function(i1){
            vec1 <- get_vector_from_df_row(df1, vector_col1, row = i1)
            
            df_sim_one_word <- sapply(1:nrow(df2), function(i2){
                                                         vec2 <- get_vector_from_df_row(df2, vector_col2, row = i2)
                                                         compute_cosin(vec1, vec2)
                                                         }) 
            
            df_sim_one_word <- as.data.frame(matrix(df_sim_one_word, nrow = 1))
            colnames(df_sim_one_word) <- as.character(df2[[var_key2]])
            
            df_sim_one_word <- cbind(df1[i1, var_key1, with = FALSE], df_sim_one_word)
            
            
            print(paste0(Sys.time(), " - computed ", i1, " word: ", df1[i1, , ][[var_key1]]))
            
            return(df_sim_one_word)
        })
        
        similarities <- rbindlist(similarities)
        
    } else {
        stop("ERROR computing similarity matrix: vector with different dimensions")
    }
    
    
    #### ...save dataset as rds file...
    if (!is.na(file_save_rds)){
        tryCatch({
            print("saving dataset")
            saveRDS(similarities, paste0(file_save_rds))
        },
        error = function(err){
            stop("ERROR saving dataset")
        })
    } 
    
    
    return (similarities)
}
