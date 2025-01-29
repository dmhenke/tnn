# Load R libs ####
library(arrow)
library(data.table)
library(ggplot2)
library(keras)
library(pROC)
library(ranger)


# Load metadata ####
meta <- fread(
  "/Users/lukas/OneDrive/Miko/THINC/projects/recursion/data/rxrx3_core_metadata.csv")
rownames(meta) <- meta$well_id


# Load embedding ####
tmp <- read_parquet(
  "/Users/lukas/OneDrive/Miko/THINC/projects/recursion/data/open_phenom_embeddings.parquet")
dat <- data.matrix(tmp[, -1])
rownames(dat) <- tmp$well_id

# scaled <- apply(dat, 2, function(x) (x - mean(x)) / sd(x))
# dat <- scaled

meta <- meta[match(rownames(dat), meta$well_id), ]


# Define triplets ####
meta_tmp <- meta[meta$perturbation_type == "CRISPR", ]
dat_tmp <- dat[meta$perturbation_type == "CRISPR", ]

meta_tmp$condition <- paste(meta_tmp$experiment_name, meta_tmp$plate)
asplit <- split(1:nrow(meta_tmp), meta_tmp$condition)

dat_subtract_ctrl <- do.call(rbind, lapply(asplit, function(x){
  ctrl <- grep("control", meta_tmp$treatment[x])
  ctrl_avg <- colMeans(dat_tmp[x[ctrl], ])
  
  dat_tmp[x, ] - ctrl_avg
}))

ok <- which(meta_tmp$gene != "EMPTY_control")

x_train <- dat_subtract_ctrl[ok, ]
meta_tmp <- meta_tmp[ok, ]

genes <- unique(meta_tmp$gene)

r <- sample(genes, 10)
test <- which(meta_tmp$gene %in% r)
train <- setdiff(1:nrow(meta_tmp), test)

meta_test <- meta_tmp[test, ]
x_test <- x_train[test, ]

meta_train <- meta_tmp[train, ]
x_train <- x_train[train, ]

asplit <- split(1:nrow(meta_train), meta_train$gene)

triplets_global <- lapply(asplit, function(x){
  pos <- x
  neg <- setdiff(1:nrow(meta_train), pos)
  
  list(positives = pos, negatives = neg)
})
names(triplets_global) <- names(asplit)


# Define model ####
margin = 1
layer_dense = 64
dims = ncol(dat)

# Set up shared layers:
dense1 <- layer_dense(units = 200, activation="relu")
dense2 <- layer_dense(units = 100, activation="relu")
embed <- layer_dense(units = layer_dense)

# Set up the triple inputs:
anchor_input <- layer_input(shape = dims, name = 'anchor')
positive_input <- layer_input(shape = dims, name = 'positive')
negative_input <- layer_input(shape = dims, name = 'negative')

# Three encoders, shared weights:
encoded_anchor <- anchor_input %>% layer_dropout(0.25) %>%
  dense1() %>% layer_dropout(0.25) %>%
  dense2() %>% 
  embed()
encoded_positive <- positive_input %>% layer_dropout(0.25) %>%
  dense1() %>% layer_dropout(0.25) %>%
  dense2() %>% 
  embed()
encoded_negative <- negative_input %>% layer_dropout(0.25) %>%
  dense1() %>% layer_dropout(0.25) %>%
  dense2() %>% 
  embed()

# Squared Euclidean distance for Anchor-Positive pair:
DAP <-
  list(encoded_anchor, encoded_positive) %>%
  layer_lambda(function(x) {
    c(encoded_anchor, encoded_positive) %<-% x
    k_sum(k_square(encoded_anchor - encoded_positive),
          axis = 2, keepdims = TRUE)
  },
  name = 'DAP_loss')

# Squared Euclidean distance for Anchor-Negative pair:
DAN <-
  list(encoded_anchor, encoded_negative) %>%
  layer_lambda(function(x) {
    c(encoded_anchor, encoded_negative) %<-% x
    k_sum(k_square(encoded_anchor - encoded_negative),
          axis = 2, keepdims = TRUE)
  },
  name = 'DAN_loss')

# Concatenate the two Euclidean distances, output is size [samples, 2]
final_layer <- layer_concatenate(list(DAP, DAN))

# Construct the model
model <- keras_model(
  inputs = list(anchor_input, positive_input, negative_input),
  outputs = final_layer
)

triplet_loss <- function(y_true, y_pred) {
  k_maximum(x = y_pred[, 1] - y_pred[, 2] + margin, y = 0)
}

# Compile model
model %>% compile(optimizer = 'rmsprop', loss = triplet_loss)


# Train #####
global_loss <- c()
b <- list()

img <- array_reshape(
  x_train, 
  dim = c(nrow(x_train), ncol(x_train)))

lapply(1:100, function(k){
  
  print(k)
  
  anchor <- img
  
  triplets <- triplets_global[meta_train$gene]
  
  ok <- unlist(lapply(triplets, function(x) sample(x$positives, 1)))
  positive <- img[ok, ]
  
  ok <- unlist(lapply(triplets, function(x) sample(x$negatives, 1)))
  negative <- img[ok, ]
  
  b <- list()
  b$anchor <- anchor
  b$positive <- positive
  b$negative <- negative
  
  h <- keras::fit(
    model,
    x = unname(b), y = array(0, c(nrow(b[[1]]), 1)),
    epochs = 1, batch_size = 264, shuffle = T)
  
  global_loss <<- c(global_loss, h$metrics$loss)
})

plot(global_loss)


# Get embedding for original matrix ####
as_keras <- function(matr){
  array_reshape(
    matr, 
    dim = c(nrow(matr), ncol(matr)))
}
encoder <- keras_model(
  inputs = anchor_input,
  outputs = encoded_anchor
)
img <- as_keras(dat)
tnn_embedding <- predict(encoder, img)


# Save as parquet ####
write_parquet(
  data.frame(tnn_embedding), 
  "/Users/lukas/OneDrive/Miko/THINC/projects/recursion/tnn_embeddings.parquet") 
