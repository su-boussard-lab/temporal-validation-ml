# Creates the boxplots in Figure 6b
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

# ------------  POINT-WISE REMOVAL, RDE - DATA VALUATION ------------
methods = c('Loo', 'Banzhaf', 'Oob', 'KNNShapley')

for (method in methods) {
  path = paste0('data/processed/RDE_RF_Single_shapley_performance_', method, '_bootstrapped.csv')
  df <- read.csv(path)
  
  # Relabel column
  df$REMOVAL <- gsub("HIGH_MONTHLY", "Remove High Value Points", df$REMOVAL)
  df$REMOVAL <- gsub("LOW_MONTHLY", "Remove Low Value Points", df$REMOVAL)
  
  if (method == 'Banzhaf') {method.label = "Data-Banzhaf"}
  if (method == 'Loo') {method.label = "LeaveOneOut"}
  if (method == 'Oob') {method.label = "Data Out-of-Bag (Oob)"}
  if (method == 'KNNShapley') {method.label = "KNNShapley"}
  
  RDE_plot <- ggplot(df, aes(x = Active_Samples, y = AUROC, color = REMOVAL)) +
    geom_line(size = 0.8) + 
    geom_point(size = 0.9) + 
    geom_ribbon(aes(ymin = AUROC_LOW, ymax = AUROC_HIGH, fill = REMOVAL), alpha = 0.2) + 
    geom_vline(xintercept = 10453, linetype = "dashed", color = "darkgrey") +  
    scale_color_manual(values = c("#2D708EFF", "#F8766D"), name = "REMOVE") + 
    scale_fill_manual(values = c("#2D708EFF", "#F8766D"), name = "REMOVE") + 
    ylim(0.5, 0.80) +  # Set y-axis limits
    scale_x_reverse(name = "Number of Training Samples") +
    theme_light() +
    labs(x = "Number of Training Samples", y = "AUROC", color = NULL, title = method.label) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11), 
      axis.text.y = element_text(size = 12), 
      legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),  
      legend.box.background = element_rect(color = NA), 
      plot.title = element_text(hjust = 0.5),
      legend.text = element_text(size = 12),  
      axis.title.x = element_text(size = 12), 
      axis.title.y = element_text(size = 12),  
      legend.spacing.y = unit(0.0, 'cm'),
      legend.position = 'top' 
    ) +
    guides(fill = guide_legend(nrow = 2, keywidth = 0.8, keyheight = 0.8))
  
  store_path = paste0('results/figure_s2/figure_s2_RDE_RF_pointwise_100_', method, '.pdf')
  ggsave(store_path, plot = RDE_plot, width = 6, height = 5)  
}