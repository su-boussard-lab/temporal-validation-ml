# Creates the boxplots in Figure 6b
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)
 
# Read in data
dataval.df <- read.csv("data/processed/data_values_1020_on_21_s42_knn_dataoob.csv")

# Select DataOob and KNNShapley
df.oob <- dataval.df %>% select(index, data_values, YEAR) %>%
  filter(index == 'DataOob(num_models=100)') %>%
  mutate(index = ifelse(index == "DataOob(num_models=100)", 'DataOob', NA)) %>% 
  mutate(rank_value = rank(data_values))

df.KNN <- dataval.df %>% select(index, data_values, YEAR) %>%
  filter(index == 'KNNShapley(k_neighbors=100)') %>%
  mutate(index = ifelse(index == 'KNNShapley(k_neighbors=100)', 'KNNShapley', NA)) %>% 
  mutate(rank_value = rank(data_values))

# Function to bootstrap the mean
bootstrap_mean <- function(data, indices) {
  return(mean(data[indices]))
}

# Compute mean and bootstrap CI for each month
mean.CI.KNN <- df.KNN %>%
  group_by(YEAR) %>%
  summarise(
    mean_value = mean(rank_value),
    lower_CI = quantile(replicate(1000, mean(sample(rank_value, replace = TRUE))), 0.025),
    upper_CI = quantile(replicate(1000, mean(sample(rank_value, replace = TRUE))), 0.975)
  ) %>% 
  ungroup

mean.CI.oob <- df.oob %>%
  group_by(YEAR) %>%
  summarise(
    mean_value = mean(rank_value),
    lower_CI = quantile(replicate(1000, mean(sample(rank_value, replace = TRUE))), 0.025),
    upper_CI = quantile(replicate(1000, mean(sample(rank_value, replace = TRUE))), 0.975)
  ) %>% 
  ungroup

mean.CI.oob$Data_Valuation_Method <- "DataOob"
mean.CI.KNN$Data_Valuation_Method <- "KNNShapley"

df.plot.data.rank <- rbind(mean.CI.oob, mean.CI.KNN)


plot.data.values <- ggplot(df.plot.data.rank, aes(x = as.character(YEAR), y = mean_value, color = Data_Valuation_Method)) +
  geom_point(position = position_dodge(width = 0.25), size = 2) +
  geom_errorbar(aes(ymin = lower_CI, ymax = upper_CI), 
                width = 0.3, size = 0.5, 
                position = position_dodge(width = 0.25)) +
  xlab("Time") +
  scale_color_manual(values = c("#2D708EFF", "#F8766D"), name = "Data Valuation Method") + 
  ylab("Data Values (Rank Space)") +
  theme_bw() +
  labs(x = "Year", 
        y = "Mean Rank of Data Values",
        legend.position = 'top',
       title = "Yearly Averages of Data Ranks") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 13),  
        axis.text.y = element_text(size = 13),  
        legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),  
        legend.box.background = element_rect(color = NA),  
        legend.text = element_text(size = 12),  
        axis.title.x = element_text(size = 14), 
        plot.title = element_text(hjust = 0.5),
        axis.title.y = element_text(size = 14),  
        legend.spacing.y = unit(0.0, 'cm'),
        legend.position = 'top') +
  guides(color = guide_legend(nrow = 2)) 

ggsave("results/figure_6/figure_6b_point_plot_dataval_knn_oob_1020_21_new.pdf", plot = plot.data.values, width = 10, height = 4)
write.csv(df.plot.data.rank, "results/figure_6/figure_6b_point_plot_dataval_knn_oob_1020_21_new.csv")