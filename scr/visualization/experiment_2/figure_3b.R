# Plot the label evolution plots in experiment 2
library(ggridges)
library(ggplot2)
library(viridis)
library(tidyverse)
library(lubridate)

# Read in data
TTE.ACU <- read.csv('data/processed/experiment_2/time_to_event_diagnosis_acu_normalized.csv')
TTE.ACU <- TTE.ACU %>% rename(TimeToEvent = index)

df.long <- pivot_longer(TTE.ACU, cols = c(ED_VISIT_normalized, HOSPITAL_ADMISSION_normalized), names_to = "variable", values_to = "Event_count")
color.selection.3c <- c("#2D708EFF", '#F8766D')
custom_labels <- c('ED Visits', 'Hospitalizations')

plot.3b <- ggplot(df.long, aes(x = TimeToEvent, y = Event_count, fill = variable)) + 
  geom_ribbon(aes(ymin = 0, ymax = Event_count), fill = "grey", alpha = 1.0) +
  theme_light() +
  geom_smooth(method = 'loess', span = 0.3, se = FALSE, size = 0.6, aes(color = variable)) + 
  scale_color_manual(values = color.selection.3c, labels = custom_labels) + 
  scale_fill_manual(values = color.selection.3c, labels = custom_labels) +
  theme_linedraw() +
  labs(x = "Time To Event [Days]", y = "Number of ACU Events (normalized by mean)") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.y = element_text(size = 10),
  ) + 
  coord_cartesian(ylim = c(0, 3))

ggsave("results/figure_3/figure_3b_time_to_event_diagnosis_acu_normalized.pdf", plot = plot.3b, width = 7, height = 4) 