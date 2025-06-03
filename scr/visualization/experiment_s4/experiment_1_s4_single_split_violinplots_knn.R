rm(list=ls())

library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

results <- data.frame()

for (k in c(5, 15, 100, 1000)) {
  result.k <- NULL
  base.path <- "results/performance_models/experiment_1/"
  train.path <- paste0(base.path, "Exp_1a_b_SingleSplit_knn", k, "_1018_1922_s42_train_period.csv")
  test.path <- paste0(base.path, "Exp_1a_b_SingleSplit_knn", k, "_1018_1922_s42_test_period.csv")
  
  df.train.results <- read.csv(train.path)
  df.test.results <- read.csv(test.path)
  
  result.k <- rbind(df.train.results, df.test.results)
  result.k$n_KNN <- k
  results <- rbind(results, result.k)
}

# Group by k and models
results$n_KNN <- factor(results$n_KNN, levels = c("5", "15", "100", "1000"))
results <- results %>% mutate(Model = if_else(Model == "cv_fit_GBM", "XGBoost",
                                              if_else(Model == "cv_fit_LASSO", "LASSO",
                                                      if_else(Model == "cv_fit_RF", "Random Forest", Model))))

results.train.years <- results %>% filter(Test_Years == "2010-2018") %>% 
  mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps)) %>% 
  filter(Model == 'XGBoost' | Model == "Random Forest" | Model == "LASSO")

results.test.years <- results %>% filter(Test_Years == "2019-2022") %>% 
  mutate(AUROC_bootstraps = str_remove_all(AUROC_bootstraps, "\\[|\\]")) %>%  separate_rows(AUROC_bootstraps, sep = ",") %>% mutate(AUROC_bootstraps = as.numeric(AUROC_bootstraps)) %>% 
  filter(Model == 'XGBoost' | Model == "Random Forest" | Model == "LASSO")


# --------- Retrospective/Training Period ---------
plot.AUROC.train.period <- ggplot(results.train.years, aes(x = Model, y = AUROC_bootstraps, fill = n_KNN)) + 
  geom_violin(trim = FALSE, alpha = 1.0, scale = "width",  width = 0.6, position = position_dodge(width = 0.7)) +
  geom_point(aes(y = AUROC), stat = "identity", position = position_dodge(width = 0.7),
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("5" = "#F8766D", "15" = "#2D708EFF", "100" = "#00BFC4", '1000'="grey"))+
  labs(y = "AUROC", title = "Retrospective Training and Test Sets (KNN−Imputation)") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.83)) +  
  theme(
    title = element_text(hjust = 0.5, size = 16),  
    axis.text.x = element_text(hjust = 0.5, size = 16),  
    axis.text.y = element_text(size = 16),  
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1), 
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 16),  
    axis.title.x = element_text(size = 16),  
    axis.title.y = element_text(size = 16), 
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'right'
  ) +
  guides(fill = guide_legend(nrow = 4, keywidth = 0.5, keyheight = 0.5)) + 
  theme(panel.grid.major = element_blank())

ggsave("results/figure_s4/S4_violinplot_knn_imputation_train_test_period_1018_new.pdf", plot = plot.AUROC.train.period, width = 12, height = 5)

# --------- Prospective Test Period ---------
plot.AUROC.test.period <- ggplot(results.test.years, aes(x = Model, y = AUROC_bootstraps, fill = n_KNN)) + 
  geom_violin(trim = FALSE, alpha = 1.0, scale = "width",  width = 0.6, position = position_dodge(width = 0.7)) +
  geom_point(aes(y = AUROC), stat = "identity", position = position_dodge(width = 0.7),
             color = "white", size = 3.0, alpha = 0.9) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), width = 0.05, position = position_dodge(width = 0.7), color = "white") +
  scale_fill_manual(values = c("5" = "#F8766D", "15" = "#2D708EFF", "100" = "#00BFC4", '1000'="grey"))+
  labs(y = "AUROC", title = "Retrospective Training and Prospective Test Sets (KNN−Imputation)") +
  theme_classic() +
  coord_cartesian(ylim = c(0.75, 0.82)) +  
  theme(
    title = element_text(hjust = 0.5, size = 16),  
    axis.text.x = element_text(hjust = 0.5, size = 16),  
    axis.text.y = element_text(size = 16),  
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1), 
    legend.box.background = element_rect(color = NA), 
    legend.text = element_text(size = 16),  
    axis.title.x = element_text(size = 16),  
    axis.title.y = element_text(size = 16), 
    legend.spacing.y = unit(0.0, 'cm'),
    legend.position = 'right'
  ) +
  guides(fill = guide_legend(nrow = 4, keywidth = 0.5, keyheight = 0.5)) + 
  theme(panel.grid.major = element_blank())

ggsave("results/figure_s4/S4_violinplot_knn_imputation_train_1018_test_1922_new.pdf", plot = plot.AUROC.test.period, width = 12, height = 5)
