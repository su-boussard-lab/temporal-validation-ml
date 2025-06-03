# Plot heatmap in Figure 6a
rm(list = ls())
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyr)
library(RColorBrewer)
library(tidyverse)
library(lubridate)
library(tsibble)

# Data values
dataval.df <- read.csv('data/processed/data_values_1020_on_21_s42_knn_dataoob.csv')
y <- read.csv('data/raw/labels.csv')

# Select DataOob and KNNShapley
df <- dataval.df %>% select(index, data_values, PAT_DEID, YEAR) %>%
  filter(index == 'DataOob(num_models=100)' | index == 'KNNShapley(k_neighbors=100)') %>% 
  mutate(index = ifelse(index == "DataOob(num_models=100)", 'DataOob', 'KNNShapley'))

# Define helper function
scale_data <- function(x) {
  return(scale(x))
}

# Convert dates
y$CHE_TX_DATE <- as.Date(y$CHE_TX_DATE)
y$CHE_TX_DATE <- format(y$CHE_TX_DATE, "%Y-%m")

# Merge dataframes
y.select <- y %>% select(CHE_TX_DATE, PAT_DEID)
merged.df <- merge(dataval.df, y.select, by = "PAT_DEID", all.x = TRUE)

# Create results.df
result.df <- merged.df %>%
  group_by(index, CHE_TX_DATE) %>%
  summarize(mean_column1 = mean(data_values, na.rm = TRUE))

# Group and scale by data valuation method
scaled.df <- result.df  %>%
  filter(index == 'DataOob(num_models=100)' | index == 'KNNShapley(k_neighbors=100)') %>% 
  mutate(index = ifelse(index == 'DataOob(num_models=100)', "DataOob",
                        ifelse(index == 'KNNShapley(k_neighbors=100)', "KNNShapley", NA))) %>% 
  group_by(index) %>%
  mutate(across(where(is.numeric) & !all_of("CHE_TX_DATE"), scale_data)) %>%
  ungroup() %>% 
  rename(value = mean_column1)

heatmap.data <- scaled.df %>%
  pivot_longer(cols = -c(index, CHE_TX_DATE), names_to = "variable", values_to = "value") %>% 
  select(index, CHE_TX_DATE, value)

colnames(heatmap.data) <- c("INDEX", "DATE_YEAR_MONTH", "VALUE")

heatmap.dataval <- ggplot(heatmap.data, aes(x = as.Date(yearmonth(DATE_YEAR_MONTH)) , y = INDEX, fill = VALUE)) +
  geom_tile(color = "lightgrey") +
  scale_fill_gradientn(colors = brewer.pal(10, "RdYlGn"), na.value = "white", oob = scales::squish) + 
  scale_x_date(date_labels = "%Y", date_breaks = "1 year")  +
  xlab("Time") +
  ylab("Procedure") +
  ggtitle("Data Valuation") +
  theme_bw() +
  theme(plot.caption = element_text(hjust = 0),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_blank())

ggsave("results/figure_6/figure_6a_heatmap_dataval_knn_oob_1020_21_new.pdf", plot = heatmap.dataval, width = 10, height = 2)
write.csv(heatmap.data, "results/figure_6/figure_6a_heatmap_dataval_knn_oob_1020_21_new.csv")
