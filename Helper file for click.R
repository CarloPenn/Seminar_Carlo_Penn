
##### click is too large, split it into 4 datasets #####
data <- read.arff("data/dataset_.arff")

# Number of parts to split into
num_parts <- 4

# Calculate number of rows per part
rows_per_file <- ceiling(nrow(data) / num_parts)

# Split the data into smaller chunks
for (i in 1:num_parts) {
  # Calculate the row range for each chunk
  start_row <- (i - 1) * rows_per_file + 1
  end_row <- min(i * rows_per_file, nrow(data))
  
  # Create the chunk of data
  chunk <- data[start_row:end_row, ]
  
  # Save the chunk as a new ARFF file
  output_filename <- paste0("data/small_dataset_part_", i, ".arff")
  write.arff(chunk, file = output_filename)
  cat("Created:", output_filename, "\n")
}

