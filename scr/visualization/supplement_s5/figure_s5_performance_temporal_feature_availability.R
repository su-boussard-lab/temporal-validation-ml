library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)
library(cowplot)

years <- seq(2010, 2018)
result.features.online <- data.frame()

for (year in years) {
  # Combine results from all years
  results.test.set.path <- paste0("data/processed/performance_results_feature_availability/results_features_online_", year, "_1922.csv")
  df.test.results <- read.csv(results.test.set.path)
  
  df.test.results$year <- year
  result.features.online <- rbind(result.features.online, df.test.results)
}

rf.results.online <- result.features.online %>% filter(Model %in% c("cv_fit_RF"))

# plot results
plot.performance.rf <- ggplot(rf.results.online, aes(x = as.character(year), y = AUROC, color = Model)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = AUROC_Low_95, ymax = AUROC_High_95), 
                width = 0.3, size = 0.5, 
                position = position_dodge(width = 0.25)) + 
  scale_color_manual(values = c("#2D708EFF"), name = "Model") +  
  theme_bw() +
  scale_y_continuous(
    name = "AUROC",  
    limits = c(0.75, 0.82),
  ) +
  labs(title = "Model Performance Based on Features Available by Year (Random Forest)") + 
  theme(
        axis.text.y = element_text(size = 14),  
        axis.title.y = element_text(size = 14),
        axis.title.x = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 16),
       legend.position = "none") +
  guides(color = guide_legend(nrow = 2)) 

plot.features.rf <- ggplot(rf.results.online, aes(x = year)) +
  geom_point(aes(y = N_Features), size = 2, color = "slategray") +  # Dot plot
  geom_line(aes(y = N_Features), color = "slategray") +  # Line connecting the points
  scale_x_continuous(breaks = rf.results.online$year, labels = as.integer(rf.results.online$year)) +
  labs(y = "N Features") +
  scale_y_continuous(
    name = "Number of Features",   
    limits = c(800, 1100),
  ) +
  theme_minimal() +
  labs(x = "Year Cutoff for Considering Feature Availability") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 13),
        axis.text.y = element_text(size = 14),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.ticks.x = element_line(),
        panel.border = element_rect(color = "black", fill = NA), 
        panel.grid = element_blank(), 
        plot.title = element_text(hjust = 0.5, size = 16),
        legend.position = "none")

plot.combined.rf <- plot_grid(plot.performance.rf, plot.features.rf, ncol = 1,
          rel_heights = c(3, 2),
          align = "v", axis = "lr")
plot.combined.rf

ggsave('results/figure_s5/Figure_S5_performance_temporal_feature_availability.pdf', plot = plot.combined.rf, width = 8, height = 5)
