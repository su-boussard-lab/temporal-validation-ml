# This file contains the code plotting the figures in experiment 2
rm(list = ls())
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

label.df <- read.csv('data/processed/experiment_2/labels_ed_hosp_chemo.csv')

# Format date
label.df$YEAR_MONTH_ADMISSION <- as.Date(label.df$YEAR_MONTH_ADMISSION, format = "%Y-%m-%d")

# Prepare dataframe
ratio.acu.initiation <- label.df %>% 
  group_by(YEAR_MONTH_ADMISSION) %>% 
  summarise(ratio = sum(Count[LABEL_TYPE == "ED" | LABEL_TYPE == "HOSPITAL_ADMISSION"]) / sum(Count[LABEL_TYPE == "TOTAL_CHEMOINIT"])) %>%
  filter(YEAR_MONTH_ADMISSION <= '2022-06-01')

color.selection <- c("#2D708EFF", '#F8766D')
custom_labels <- c('ED Visits', 'Hospitalizations')

# Save data for plotting
write.csv(ratio.acu.initiation, "data/processed/experiment_2/experiment_2_labels_hosp_chemo_ratio.csv")

plot.3a.bottom <- ggplot(ratio.acu.initiation, aes(x = YEAR_MONTH_ADMISSION, y = ratio)) + 
  geom_smooth(method = 'loess', span = 0.1, se=FALSE, size = 0.5, color ="#F8766D") + 
  scale_color_manual(values = color.selection, labels = custom_labels) +  
  scale_x_date(date_labels = "%Y", date_breaks = "1 year") +  
  theme_linedraw() +
  labs(x = "Time", y = "ACU Events", color = NULL) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1), 
  ) + 
  ylim(0.05, 0.35) +
  theme(
    legend.box.margin = margin(0.1, 0.1, 0.1, 0.1), 
    legend.box.background = element_rect(color = NA),  
    legend.text = element_text(size = 8), 
    axis.text.x = element_text(angle = 45, hjust = 1), 
    legend.spacing.y = unit(0.0, 'cm'),
    axis.title.y = element_text(size = 10) 
  ) +
  guides(color = guide_legend(nrow = 1))

ggsave("results/figure_3/figure_3a_bottom_ratio_acu_therapy_initiation.pdf", plot = plot.3a.bottom, width = 10, height = 2) 
