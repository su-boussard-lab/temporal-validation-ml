# Plot Figure S2
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

base.path <-  "results/performance_models/experiment_4/Exp_4c_singlesplit_final_" 

# Full data
ay.af <- read.csv(paste0(base.path, "full_data_mean_1020_22_s42_test_period.csv"))
ay.af$config <- "AY+AF"

# All years, reduced features
ay.rf <- read.csv(paste0(base.path, "feature_reduction_mean_1020_22_s42_test_period.csv"))
ay.rf$config <- "AY+RF"

# Reduced years, all features
ry.af <- read.csv(paste0(base.path, "all_features_data_reduction_mean_1020_22_s42_test_period.csv"))
ry.af$config <- "RY+AF"

# Reduced years, reduced features
ry.rf <- read.csv(paste0(base.path, "feature_reduction_data_reduction_mean_1020_22_s42_test_period.csv"))
ry.rf$config <- "RY+RF"

# Combine all dataframes
df.combined <- rbind(ay.af, ay.rf, ry.af, ry.rf) 

df.AUROC <- df.combined %>% 
  filter(Model == 'cv_fit_GBM' | Model == "cv_fit_RF" | Model == "cv_fit_LASSO") %>% 
  mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  
  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps)) 

df.AUPRC <- df.combined %>% 
  filter(Model == 'cv_fit_GBM' | Model == "cv_fit_RF" | Model == "cv_fit_LASSO") %>% 
  mutate(AUPRC_bootstraps = str_remove_all(AUPRC_bootstraps, "\\[|\\]")) %>%  
  separate_rows(AUPRC_bootstraps, sep = ",") %>% mutate(AUPRC_bootstraps = as.numeric(AUPRC_bootstraps)) 

# Define legends and labels
legend.labels <- c("AY+AF" = "Full Time Period (2010-20) + Feature Set", 
                   "AY+RF" = "Full Time Period + Red. Feature Set",
                   "RY+AF" = "Pruned Time Period + Full Feature Set",
                   "RY+RF" = "Pruned Time Period + Red. Feature Set")
custom.labels <- c("cv_fit_GBM" = "XGBoost", "cv_fit_LASSO" = "LASSO", "cv_fit_RF" = "Random Forest")


plot.AUROC <- ggplot(df.AUROC, aes(x = Model, y = AUROC_bootstraps, fill = config)) + 
  geom_violin(trim = FALSE, position = position_dodge(width = 0.7), alpha = 0.9, scale = "width",  width = 0.6) +
  geom_point(aes(y = AUROC), stat = "identity", position = position_dodge(width = 0.7),
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("AY+AF" = "#F8766D", "AY+RF" = "#2D708EFF", "RY+AF" = "#00BFC4", 'RY+RF'="grey"),
                    labels = legend.labels) +
  labs(y = "AUROC", title = "Model Comparison: post Data and Feature Reduction") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.85)) +
  scale_x_discrete(labels = custom.labels) + 
  theme(
    title = element_text(size = 14),
    axis.text.x = element_text(hjust = 0.5, size = 14),  
    axis.text.y = element_text(size = 14),
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1), 
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 14),
    axis.title.x = element_text(size = 14),  
    axis.title.y = element_text(size = 14),  
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank()) 

ggsave('results/figure_s3/s3_violinplot_single_split_postRFE_postRDE_AUROC_full_final.pdf', plot = plot.AUROC, width = 10, height = 4) 

plot.AUPRC <- ggplot(df.AUPRC, aes(x = Model, y = AUPRC_bootstraps, fill = config)) + 
  geom_violin(trim = FALSE, position = position_dodge(width = 0.7), alpha = 0.9, scale = "width",  width = 0.6) +
  geom_point(aes(y = AUPRC), stat = "identity", position = position_dodge(width = 0.7),
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUPRC_Low_95, ymax = AUPRC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("AY+AF" = "#F8766D", "AY+RF" = "#2D708EFF", "RY+AF" = "#00BFC4", 'RY+RF'="grey"),
                    labels = legend.labels) +
  labs(y = "AUPRC", title = "Model Comparison: post Data and Feature Reduction") +
  theme_classic() +
  coord_cartesian(ylim = c(0.4, 0.6)) +
  scale_x_discrete(labels = custom.labels) + 
  theme(
    title = element_text(size = 14),
    axis.text.x = element_text(hjust = 0.5, size = 14),  
    axis.text.y = element_text(size = 14),
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1), 
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 14),
    axis.title.x = element_text(size = 14),  
    axis.title.y = element_text(size = 14),  
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank()) 

ggsave('results/figure_s3/s3_violinplot_single_split_postRFE_postRDE_AUPRC_full_final.pdf', plot = plot.AUPRC, width = 10, height = 4)