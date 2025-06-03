rm(list=ls())

library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

results <- data.frame()
for (k in c(5, 15, 100, 1000)) {
  result.k <- NULL
  test.path <- paste0("results/performance_models/experiment_1/Exp_1a_b_SingleSplit_knn", k, "_1018_1922_s42_test_period.csv")
  train.path <- paste0("results/performance_models/experiment_1/Exp_1a_b_SingleSplit_knn", k, "_1018_1922_s42_train_period.csv")
  
  df.train.results <- read.csv(train.path)
  df.train.results <- df.train.results %>%
    mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps))   
  
  df.test.results <- read.csv(test.path)
  df.test.results <- df.test.results %>%
    mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps))   
  
  result.k <- rbind(df.train.results, df.test.results)
  result.k$n_KNN <- k
  results <- rbind(results, result.k)
}

# Group by k and models
results$n_KNN <- factor(results$n_KNN, levels = c("5", "15", "100", "1000"))
  

results <- results %>% 
  filter(Model == 'cv_fit_GBM' | Model == "cv_fit_RF" | Model == "cv_fit_LASSO") %>% 
  mutate(Model = if_else(Model == "cv_fit_GBM", "XGBoost",
                                              if_else(Model == "cv_fit_LASSO", "Lasso",
                                                      if_else(Model == "cv_fit_RF", "Random Forest", Model))))
results.train.years <- results %>% filter(Test_Years == "2010-2018") 
results.test.years <- results %>% filter(Test_Years == "2019-2022")

# --------- Retrospective/Training Period ---------
plot.AUROC.train.period <- ggplot(results.train.years, aes(x = Model, y = AUROC_bootstraps, fill = n_KNN)) + 
  geom_violin(trim = FALSE, position = position_dodge(width = 0.7), alpha = 0.9, scale = "width",  width = 0.6) +
  geom_point(aes(y = AUROC), stat = "identity", position = position_dodge(width = 0.7),
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  labs(y = "AUROC on Test Set", title = "Retrospective Training and Test Set") +
  scale_fill_manual(values = c("5" = "#F8766D", "15" = "#2D708EFF", "100" = "#00BFC4", '1000'="grey"))+
  labs(y = "AUROC", title = "Retrospective Training and Test Sets (KNN-Imputation)") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.825)) + 
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
    legend.position = 'right'
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank()) 

ggsave('results/figure_s4/S4_violinplot_knn_imputation_train_test_period_1018.pdf', plot = plot.AUROC.train.period, width = 12, height = 5)

# --------- Prospective Test Period ---------
plot.AUROC.test.period <- ggplot(results.test.years, aes(x = Model, y = AUROC_bootstraps, fill = n_KNN)) + 
  geom_violin(trim = FALSE, position = position_dodge(width = 0.7), alpha = 0.9, scale = "width",  width = 0.6) +
  geom_point(aes(y = AUROC), stat = "identity", position = position_dodge(width = 0.7),
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  labs(y = "AUROC on Test Set", title = "Retrospective Training and Test Set") +
  scale_fill_manual(values = c("5" = "#F8766D", "15" = "#2D708EFF", "100" = "#00BFC4", '1000'="grey"))+
  labs(y = "AUROC", title = "Retrospective Training Set and Prospective Test Set (KNN-Imputation)") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.825)) + 
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
    legend.position = 'right'
  ) +
  guides(fill = guide_legend(nrow = 3, keywidth = 0.8, keyheight = 0.8)) + 
  theme(panel.grid.major = element_blank()) 

ggsave('results/figure_s4/S4_violinplot_knn_imputation_train_1018_test_1922.pdf', plot = plot.AUROC.test.period, width = 12, height = 5)
