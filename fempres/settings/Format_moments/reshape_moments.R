library(tidyverse)
library(magrittr)
library(yaml)


library(dplyr)  
# Create the mapping
reverse_mapping <- c(
  "Male, no gender" = "tau_01", 
  "Male, no gender" = "tau_00", 
  "Male, gender" = "tau_11", 
  "Male, gender" = "tau_10",
  "Female, no gender" = "tau_21", 
  "Female, no gender" = "tau_20", 
  "Female, gender" = "tau_31", 
  "Female, gender" = "tau_30"
)

df <- as_tibble(read.csv("moments.csv", header = T))%>%
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
  mutate(
    gender = ifelse(!is.na(gender), as.integer(gender), NA_integer_),
    group = reverse_mapping[treatm]  # Here we use the reverse mapping to create the group column
  ) %>%
  pivot_wider(
    names_from = variable,
    values_from = "value"
  )

#Export as csv
df %>%
  write.csv(
    file = "moments_clean.csv",
    na = "0", #Replace NAs with 0, comment out if not needed
    row.names = FALSE,
  )

df$av_knowledge_cumul <- NA
df$ac_CW_cumul <- NA

varlist <-colnames(df)

yaml_content <- as.yaml(list(column_names = varlist))
write(yaml_content, file = "column_names.yml")
