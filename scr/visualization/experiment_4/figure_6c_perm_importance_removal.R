# Creates the line plots for feature importance in Figure 6c
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

# ------ 1st Cycle: Feature removal ------
base.path <- "results/performance_models/experiment_4/RFE_permutation_importance/"
RFE.df <- read.csv(paste0(base.path, 'RFE_permutation_importance_step_size_10_1020_21_c1_351.csv'))

RFE.df.random <- read.csv(paste0(base.path,'RFE_permutation_importance_step_size_10_1020_21_c1_random.csv'))

RFE.df$SELECTION <- 'Permutation Importance-based Selection'
RFE.df.random$SELECTION <- 'Random Selection'
df.shap.c1 <- rbind(RFE.df, RFE.df.random)

RFE.plot.c1 <- ggplot(df.shap.c1, aes(x = Active_Features, y = AUROC, color = SELECTION)) +
  geom_line(size = 0.8) +  
  geom_point(size = 0.9) +  
  geom_ribbon(aes(ymin = AUROC_LOW, ymax = AUROC_HIGH, fill = SELECTION), alpha = 0.2) +  # Add C.I.
  geom_vline(xintercept = (df.shap.c1$Active_Features[2]-351), linetype = "dashed", color = "darkgrey") +  
  scale_color_manual(values = c("#2D708EFF", "#F8766D"), name = "SELECTION") +  # Custom colors for lines
  scale_fill_manual(values = c("#2D708EFF", "#F8766D"), name = "SELECTION") +  # Custom colors for ribbons
  ylim(0.6, 0.82) +  
  scale_x_reverse(name = "Number of Training Samples") +#, breaks = c(1000, 750, 500, 250, 100, 0)) +  # Reverse the x-axis
  theme_linedraw() +
  labs(x = "Number of Training Samples", y = "AUROC", color = NULL, title = "Feature removal: 1st Cycle") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11), 
    plot.title = element_text(hjust = 0.5),
    axis.text.y = element_text(size = 12), 
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),  
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 12),  
    axis.title.x = element_text(size = 12),  
    axis.title.y = element_text(size = 12),  
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 2, keywidth = 0.8, keyheight = 0.8))

store.path.RFE.c1 = paste0('results/figure_6/figure_6c_RFE_RF_Features_Permutation_1020_21_nfeat_351_final.pdf')
ggsave(store.path.RFE.c1, plot = RFE.plot.c1, width = 7, height = 7)  

# ------ 2nd Cycle: Feature removal ------
RFE.df <- read.csv(paste0(base.path, 'RFE_permutation_importance_step_size_10_1020_21_c2_183.csv'))
RFE.df.random <- read.csv(paste0(base.path, 'RFE_permutation_importance_step_size_10_1020_21_c2_random.csv'))

RFE.df$SELECTION <- 'Permutation Importance-based Selection'
RFE.df.random$SELECTION <- 'Random Selection'
df.shap.c2 <- rbind(RFE.df, RFE.df.random)

RFE.plot.c2 <- ggplot(df.shap.c2, aes(x = Active_Features, y = AUROC, color = SELECTION)) +
  geom_line(size = 0.8) +  
  geom_point(size = 0.9) +  
  geom_ribbon(aes(ymin = AUROC_LOW, ymax = AUROC_HIGH, fill = SELECTION), alpha = 0.2) +  # Add C.I.
  geom_vline(xintercept = (df.shap.c2$Active_Features[2]-183), linetype = "dashed", color = "darkgrey") +  
  scale_color_manual(values = c("#2D708EFF", "#F8766D"), name = "SELECTION") +  # Custom colors for lines
  scale_fill_manual(values = c("#2D708EFF", "#F8766D"), name = "SELECTION") +  # Custom colors for ribbons
  ylim(0.6, 0.82) +  
  scale_x_reverse(name = "Number of Training Samples") +#, breaks = c(1000, 750, 500, 250, 100, 0)) +  # Reverse the x-axis
  theme_linedraw() +
  labs(x = "Number of Training Samples", y = "AUROC", color = NULL, title = "Feature removal: 2nd Cycle") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11), 
    plot.title = element_text(hjust = 0.5),
    axis.text.y = element_text(size = 12), 
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),  
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 12),  
    axis.title.x = element_text(size = 12),  
    axis.title.y = element_text(size = 12),  
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 2, keywidth = 0.8, keyheight = 0.8))

store.path.RFE.c2 = paste0('results/figure_6/figure_6c_RFE_RF_Features_Permutation_1020_21_c_2_nfeat_183.pdf')
ggsave(store.path.RFE.c2, plot = RFE.plot.c2, width = 7, height = 7)  

# ------ 3rd Cycle: Feature removal ------
RFE.df <- read.csv(paste0(base.path, 'RFE_permutation_importance_step_size_10_1020_21_c3_94.csv'))
RFE.df.random <- read.csv(paste0(base.path, 'RFE_permutation_importance_step_size_10_1020_21_c3_random.csv'))

RFE.df$SELECTION <- 'Permutation Importance-based Selection'
RFE.df.random$SELECTION <- 'Random Selection'
df.shap.c3 <- rbind(RFE.df, RFE.df.random)

RFE.plot.c3 <- ggplot(df.shap.c3, aes(x = Active_Features, y = AUROC, color = SELECTION)) +
  geom_line(size = 0.8) +  
  geom_point(size = 0.9) +  
  geom_ribbon(aes(ymin = AUROC_LOW, ymax = AUROC_HIGH, fill = SELECTION), alpha = 0.2) +  # Add C.I.
  geom_vline(xintercept = (df.shap.c3$Active_Features[2]-94), linetype = "dashed", color = "darkgrey") +  
  scale_color_manual(values = c("#2D708EFF", "#F8766D"), name = "SELECTION") +  # Custom colors for lines
  scale_fill_manual(values = c("#2D708EFF", "#F8766D"), name = "SELECTION") +  # Custom colors for ribbons
  ylim(0.6, 0.82) +  
  scale_x_reverse(name = "Number of Training Samples") +#, breaks = c(1000, 750, 500, 250, 100, 0)) +  # Reverse the x-axis
  theme_linedraw() +
  labs(x = "Number of Training Samples", y = "AUROC", color = NULL, title = "Feature removal: 3rd Cycle") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11), 
    plot.title = element_text(hjust = 0.5),
    axis.text.y = element_text(size = 12), 
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),  
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 12),  
    axis.title.x = element_text(size = 12),  
    axis.title.y = element_text(size = 12),  
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 2, keywidth = 0.8, keyheight = 0.8))

store.path.RFE.c3 = paste0('results/figure_6/figure_6c_RFE_RF_Features_Permutation_1020_21_c_3_nfeat_94.pdf')
ggsave(store.path.RFE.c3, plot = RFE.plot.c3, width = 7, height = 7)  