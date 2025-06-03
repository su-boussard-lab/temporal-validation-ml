# This file contains the code for plotting the figures in experiment 1 (Figure 2)
rm(list=ls())
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

# ---- Load performance on prospective test set ----
df.1a.retro.pro <- read.csv("results/performance_models/experiment_1/experiment_1a_b_final_singlesplit_mean_1018_1922_s42_train_period.csv")
df.1a.retro.pro$Configuration <- 'Retrospective Train and Test Set'
df.1a.retro.pro <- df.1a.retro.pro %>%
  mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps))   

df.1b.pro <- read.csv("results/performance_models/experiment_1/experiment_1a_b_final_singlesplit_mean_1018_1922_s42_test_period.csv")
df.1b.pro$Configuration <- 'Retrospective Train and Prospective Test Set'
df.1b.pro <- df.1b.pro %>%
  mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps))   

df.1c <- read.csv("results/performance_models/experiment_1/experiment_1c_final_singlesplit_mean_1922_1922_s42_train_period.csv")
df.1c$Configuration <- 'Prospective Train and Test Set'
df.1c <- df.1c %>%
  mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps))   

df.1d <- read.csv("results/performance_models/experiment_1/experiment_1d_final_singlesplit_mean_1022_1022_s42_train_period.csv")
df.1d$Configuration <- 'Full Cohort'
df.1d <- df.1d %>%
  mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps))   

df.combined <- rbind(df.1a.retro.pro, df.1b.pro)
df.combined <- rbind(df.combined, df.1c)
df.combined <- rbind(df.combined, df.1d)

# Select GBM, RF and Lasso for comparison
df.combined <- df.combined %>% 
  filter(Model == 'cv_fit_GBM' | Model == "cv_fit_RF" | Model == "cv_fit_LASSO")

# ----- Plot Figure 2a - retrospective plot ----
df.fig.2a.retro <- df.combined %>% filter(Configuration == "Retrospective Train and Test Set")
plot.fig.2a.retro <- ggplot(df.fig.2a.retro, aes(x = Model, y = AUROC_bootstraps, fill = Model)) + 
  geom_violin(trim = FALSE, alpha = 1.0, scale = "width",  width = 0.8) +
  geom_point(aes(y = AUROC), stat = "identity", position = 'identity',
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("cv_fit_RF" = "#F8766D", "cv_fit_GBM" = "#2D708EFF", "cv_fit_LASSO" = "#99BFC4"),
                    labels = c("Random Forest", "Lasso", "XGBoost")) +
  labs(y = "AUROC on Test Set", title = "Retrospective Training and Test Set") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.83)) +  
  theme(
    axis.text.x = element_text(hjust = 1, size = 11),  
    axis.text.y = element_text(size = 16),  
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1), 
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 16),  
    axis.title.x = element_text(size = 16),  
    axis.title.y = element_text(size = 16), 
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top'
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank(),
        axis.ticks.x = element_blank(),  
        axis.text.x = element_blank())    

ggsave('results/figure_2/figure_violin_2a_retrospective_train_retrospective_test_1018.pdf', plot = plot.fig.2a.retro, width = 6, height = 6) 


# ---- Plot Figure 2b - prospective plot ----
df.fig.2b <- df.combined %>% filter(Configuration == "Retrospective Train and Prospective Test Set") 
plot.fig.2b <- ggplot(df.fig.2b, aes(x = Model, y = AUROC_bootstraps, fill = Model)) + 
  geom_violin(trim = FALSE, alpha = 1.0, scale = "width",  width = 0.8) +
  geom_point(aes(y = AUROC), stat = "identity", position = 'identity',
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("cv_fit_RF" = "#F8766D", "cv_fit_GBM" = "#2D708EFF", "cv_fit_LASSO" = "#99BFC4"),
                    labels = c("Random Forest", "Lasso", "XGBoost")) +
  scale_fill_manual(values = c("cv_fit_RF" = "#F8766D", "cv_fit_GBM" = "#2D708EFF", "cv_fit_LASSO" = "#99BFC4"),
                    labels = c("Random Forest", "Lasso", "XGBoost")) + 
  labs(y = "AUROC on Test Set", title = "Retrospective Training and Prospective Test Set") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.83)) +  
  theme(
    axis.text.x = element_text(hjust = 1, size = 11),  
    axis.text.y = element_text(size = 16),  
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),
    legend.box.background = element_rect(color = NA),  
    legend.text = element_text(size = 16),  
    axis.title.x = element_text(size = 16), 
    axis.title.y = element_text(size = 16), 
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank(),
        axis.ticks.x = element_blank(),  
        axis.text.x = element_blank())    

ggsave('results/figure_2/figure_violin_2b_retrospective_train_prospective_test_1018_1922.pdf', plot = plot.fig.2b, width = 6, height = 6)  # Adjust width and height as needed


# ---- Plot Figure 2c ----
df.fig.2c.prosp.train.test <- df.combined %>% filter(Configuration == "Prospective Train and Test Set")
plot.fig.2c.prosp.train.test <- ggplot(df.fig.2c.prosp.train.test, aes(x = Model, y = AUROC_bootstraps, fill = Model)) + 
  geom_violin(trim = FALSE, alpha = 1.0, scale = "width",  width = 0.8) +
  geom_point(aes(y = AUROC), stat = "identity", position = 'identity',
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("cv_fit_RF" = "#F8766D", "cv_fit_GBM" = "#2D708EFF", "cv_fit_LASSO" = "#99BFC4"),
                    labels = c("Random Forest", "Lasso", "XGBoost")) +
  scale_fill_manual(values = c("cv_fit_RF" = "#F8766D", "cv_fit_GBM" = "#2D708EFF", "cv_fit_LASSO" = "#99BFC4"),
                    labels = c("Random Forest", "Lasso", "XGBoost")) +  
  labs(y = "AUROC on Test Set", title = "Prospective Training and Test Sets") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.83)) +  
  theme(
    axis.text.x = element_text(hjust = 1, size = 11),  
    axis.text.y = element_text(size = 16),  
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),  
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 16), 
    axis.title.x = element_text(size = 16), 
    axis.title.y = element_text(size = 16),  
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank(),
        axis.ticks.x = element_blank(), 
        axis.text.x = element_blank())  

ggsave('results/figure_2/figure_violin_2c_prospective_train_test.pdf', plot = plot.fig.2c.prosp.train.test, width = 6, height = 6) 

# ---- Plot Figure 2d ----
df.fig.2d.full.cohort <- df.combined %>% filter(Configuration == "Full Cohort")
plot.fig.2d.full.cohort <- ggplot(df.fig.2d.full.cohort, aes(x = Model, y = AUROC_bootstraps, fill = Model)) + 
  geom_violin(trim = FALSE, alpha = 1.0, scale = "width",  width = 0.8) +
  geom_point(aes(y = AUROC), stat = "identity", position = 'identity',
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("cv_fit_RF" = "#F8766D", "cv_fit_GBM" = "#2D708EFF", "cv_fit_LASSO" = "#99BFC4"),
                    labels = c("Random Forest", "Lasso", "XGBoost")) +
  scale_fill_manual(values = c("cv_fit_RF" = "#F8766D", "cv_fit_GBM" = "#2D708EFF", "cv_fit_LASSO" = "#99BFC4"),
                    labels = c("Random Forest", "Lasso", "XGBoost")) +  
  labs(y = "AUROC on Test Set", title = "Training and Test Sets with Full Time Span") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.83)) +  
  theme(
    axis.text.x = element_text(hjust = 1, size = 11),  
    axis.text.y = element_text(size = 16),  
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1),  
    legend.box.background = element_rect(color = NA),  
    legend.text = element_text(size = 16), 
    axis.title.x = element_text(size = 16), 
    axis.title.y = element_text(size = 16),  
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'top' 
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank(),
        axis.ticks.x = element_blank(),   
        axis.text.x = element_blank())    

ggsave('results/figure_2/figure_violin_2d_full_time_span.pdf', plot = plot.fig.2d.full.cohort, width = 6, height = 6) 
