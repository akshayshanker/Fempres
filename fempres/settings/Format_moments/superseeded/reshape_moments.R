library(tidyverse)
library(magrittr)


library(dplyr)  


df <- as_tibble(read.csv("moments.csv", header = T))
  #Long into single column
  pivot_longer(
    cols = -c(1,2),
    names_to = "key",
    values_to = "value",
    values_drop_na = FALSE
  ) %>%
  #Separate final character and create gender column
  separate(
    key,
    into = c("variable", "gender"),
    sep = -2
  ) %>%
  mutate(gender = as.integer(substr(gender, 2, 2))) %>%
  #Recollect variable names into columns
  pivot_wider(
    names_from = variable,
    values_from = "value"
  )

#Export as csv
df %>%
  write.csv(
    file = "moments_cleaned.csv",
    na = "0", #Replace NAs with 0, comment out if not needed
    row.names = FALSE,
  )
