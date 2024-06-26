# R scripts plotting supp_Figure. 1
#Load packages
library(ggpubr)
library(ggplot2)
library(tidyverse)

# define plot path
plots_dir = "/data1/reznike/xiea1/MetabolicModel/final_results/figures/supp_fig_1/"

########## (1) supp_Fig. 1b
# Bar plots of # well-predicted metabolites in naive Lasso, MIRTH, UnitedMet -- Benchmarking
# f_test_prop=1
data_path = "/data1/reznike/xiea1/MetabolicModel/results_RNA_ccRCC/final_benchmark_bar_plot_df_final.csv"
df <- read.csv(data_path)
df <- dplyr::rename(df, metabolite = X)

# bar plot
ggplot(df, aes(x = method, fill=dataset)) +
  geom_bar(aes(y=well_predicted_percent), stat = "identity") +
  facet_wrap(~dataset, nrow=1) + 
  theme_Publication() +
  xlab("Dataset") + 
  ylab("% of well-predicted metabolites") +
  scale_fill_brewer(palette="Pastel1") 
ggsave(file.path(paste0(plots_dir,"supp_fig_1b_benchmark_well_predicted_bar_plot_100.pdf")),width = 8, height = 4) 


########## (2) supp_Fig. 1c
# Violin plots of spearman's rho of naive Lasso, MIRTH, UnitedMet -- 50% held-out Benchmarking
data_path = "/data1/reznike/xiea1/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/final_benchmark_long_table_final.csv"
df <- read.csv(data_path)
df <- dplyr::rename(df, metabolite = X)

# violin plot
ggplot(df, aes(x = method, y=rho, fill=dataset)) +
  geom_violin(width=1) +
  geom_boxplot(width=0.1, alpha=0.2) +
  facet_wrap(~dataset, nrow=1) + 
  theme_Publication() +
  xlab("Dataset") + 
  ylab("Spearman rho") +
  scale_fill_brewer(palette="Pastel1") 
ggsave(file.path(paste0(plots_dir,"supp_fig_1c_benchmark_rho_violin.pdf")),width = 8, height = 4) 

# box plot
ggplot(df, aes(x = method, y=rho, fill=dataset)) +
  geom_boxplot(fatten=2) +
  facet_wrap(~dataset, nrow=1) + 
  theme_Publication() +
  xlab("Dataset") + 
  ylab("Spearman rho") +
  scale_fill_brewer(palette="Pastel1") 
ggsave(file.path(paste0(plots_dir,"supp_fig_1c_benchmark_rho_box.pdf")),width = 8, height = 4) 


########## (3) supp_Fig. 1d
# bar plots of # well-predicted metabolites in naive Lasso, MIRTH, UnitedMet -- -- 50% held-out Benchmarking
data_path = "/data1/reznike/xiea1/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/final_benchmark_bar_plot_df_final.csv"
df <- read.csv(data_path)
df <- dplyr::rename(df, metabolite = X)

# bar plot
ggplot(df, aes(x = method, fill=dataset)) +
  geom_bar(aes(y=well_predicted_percent), stat = "identity") +
  facet_wrap(~dataset, nrow=1) + 
  theme_Publication() +
  xlab("Dataset") + 
  ylab("% of well-predicted metabolites") +
  scale_fill_brewer(palette="Pastel1") 
ggsave(file.path(paste0(plots_dir,"supp_fig_1d_benchmark_well_predicted_bar_plot_50.pdf")),width = 8, height = 4) 





theme_Publication <- function(base_size=7, family="Helvetica") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size)
    + theme(plot.title = element_text(size = rel(1.2), hjust = 0.5, family = family),
            text = element_text(family = family),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(family = family),
            axis.title.y = element_text(angle=90,vjust =2, family = family),
            axis.title.x = element_text(vjust = -0.2, family = family),
            axis.text = element_text(family = family),
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour="#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "bottom",
            legend.direction = "horizontal",
            legend.key.size= unit(0.2, "cm"),
            # legend.margin = unit(t=0, unit="cm"),
            legend.title = element_text(face="italic", family = family),
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#ffffff",fill="#ffffff"),
            strip.text = element_text()
    ))
}




