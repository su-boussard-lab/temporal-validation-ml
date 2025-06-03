# Plots the heatmap figures for 
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

# ---- Plot: Sliding Window ----
results.sw <- read.csv('data/processed/experiment_3/experiment_3a_sliding_window_predictions_rf.csv')
df.plot <- results.sw %>% 
  select(Training_Years, Test_Years, AUROC, AUROC_Low_95, AUROC_High_95) %>% 
  mutate(Test_Years = substr(Test_Years, 1, 4)) %>% 
  mutate(CI = paste0('(', round(AUROC_Low_95, 2), ",", round(AUROC_High_95, 2), ')'))

plot.title.sw <- 'Sliding Window Analysis: 3-year span'
g1 <- ggplot(df.plot, aes(x = Test_Years, y = Training_Years, fill = round(AUROC, 2))) +
  theme_bw() +
  geom_tile(position = 'identity') +
  xlab("Test Set") +
  ylab("Training Set") +
  scale_fill_viridis(limits = c(0.5, 0.8), option = "mako", discrete=FALSE, na.value = "white", oob = scales::squish) +
  ggtitle(plot.title.sw) + 
  theme(plot.caption = element_text(hjust = 0)) +
  geom_text(aes(label = round(AUROC, 2)), col = "white", size = 6) +
  theme(
    text = element_text(size=20),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 20),
    axis.text.y = (element_text(size = 20)))+ 
  labs(fill = 'AUROC')                                 
ggsave("results/figure_5/figure_5a_sliding_window.pdf", height = 10, width = 15, plot = g1)  


# ---- Plot: Real-World Progression ----
results.rwp <- read.csv('data/processed/experiment_3/experiment_3b_real_world_progression_predictions_rf.csv')
df.rwp <- results.rwp %>% 
  select(Training_Years, Test_Years, AUROC, AUROC_Low_95, AUROC_High_95) %>% 
  mutate(Test_Years = substr(Test_Years, 1, 4)) %>% 
  mutate(CI = paste0('(', round(AUROC_Low_95, 2), ",", round(AUROC_High_95, 2), ')'))

plot.title.rwp <- 'Real-World Progression Analysis - Incremental Learning'
g2 <- ggplot(df.rwp, aes(x = Test_Years, y = Training_Years, fill = round(AUROC, 2))) +
  theme_bw() +
  geom_tile() +
  xlab("Test Set") +
  ylab("Training Set") +
  scale_fill_viridis(limits = c(0.5, 0.80), option = "mako", discrete=FALSE, na.value = "white", oob = scales::squish) +
  geom_tile(color = "lightgrey") +
  ggtitle(plot.title.rwp) +
  theme(plot.caption = element_text(hjust = 0)) +
  geom_text(aes(label = round(AUROC, 2)), col = "white", size = 6) +
  theme(text=element_text(size=20),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 20),
        axis.text.y = (element_text(size = 20))) + 
  labs(fill = 'AUROC')
  
ggsave("results/figure_5/figure_5b_real_world_progression.pdf", height = 10, width = 15, plot = g2)


# ---- Plot: Retrospective Progression Analysis ----
results.rwpr <- read.csv('data/processed/experiment_3/experiment_3c_retrospective_regression_analysis_predictions_rf.csv')
df.rwpr <- results.rwpr %>% 
  select(Training_Years, Test_Years, AUROC, AUROC_Low_95, AUROC_High_95) %>% 
  mutate(Test_Years = substr(Test_Years, 1, 4)) %>% 
  mutate(CI = paste0('(', round(AUROC_Low_95, 2), ",", round(AUROC_High_95, 2), ')')) %>% 
  mutate(Training_BackTo = substr(Training_Years, 1, 4))

plot.title.rwpr <- 'Retrospective Analysis: Reverse Incremental Learning'
g3 <- ggplot(df.rwpr, aes(x = Training_BackTo, y = Test_Years, fill = round(AUROC, 2))) +
  theme_bw() +
  geom_tile() +
  xlab("Training Set") +
  ylab("Test Set") +
  scale_fill_viridis(limits = c(0.5, 0.80), option = "mako", discrete=FALSE, na.value = "white", oob = scales::squish) +
  geom_tile(color = "lightgrey") +
  scale_x_discrete(labels = function(x) paste0(x, "-(TY-1)")) +
  ggtitle(plot.title.rwpr) + 
  theme(plot.caption = element_text(hjust = 0)) +
  geom_text(aes(label = round(AUROC, 2)), col = "white", size = 6) +
  coord_flip() +
  theme(text=element_text(size=20),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 20),
        axis.text.y = (element_text(size = 20)))+ 
  labs(fill = 'AUROC')
ggsave("results/figure_5/figure_5c_retrospective_progression.pdf", height = 10, width = 15, plot = g3)





