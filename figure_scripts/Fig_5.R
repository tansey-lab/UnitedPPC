# R scripts plotting Figure. 5
plots_dir = "/data1/resnike/xiea1/MetabolicModel/final_results/figures/fig_5/"


######## (1) Fig. 5a
# Box plots of isotopologues v.s. pathological stage, showing increased TCA cycle activity is associated with disease progression
data_dir = "/data1/resnike/xiea1/MetabolicModel/final_results/data/fig_5/"
df <- read.csv(paste0(data_dir,"KIRC_isotope_stage.csv"), row.names = 1)

isotopologues <- c('Citrate.m.2', 'Succinate.m.2', 'Malate.m.2')
df_filtered <- df %>% select(one_of(c('pathologic_stage', isotopologues)))
df_long <- pivot_longer(df_filtered, cols = all_of(isotopologues), names_to = "isotopologue", values_to = "value")

# Plot boxplots and jitter points
ggplot(df_long, aes(x = pathologic_stage, y = value)) +
  geom_boxplot(aes(fill = pathologic_stage), outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  facet_wrap(~isotopologue, scales = "free_y", ncol = 3) +
  labs(title = "Isotopologue abundance by stage") +
  theme_Publication() +
  xlab("Pathologic stage") + 
  ylab("Abundance") +
  scale_fill_brewer(palette="Pastel1")  
ggsave(file.path(paste0(plots_dir,"fig_5a_isotopologue_stage.pdf")), width = 6, height = 3)

# kruskal test
print(kruskal.test(Citrate.m.2 ~ pathologic_stage, data = df_filtered)$p.value) # p = 0.0002757036
print(kruskal.test(Succinate.m.2 ~ pathologic_stage, data = df_filtered)$p.value) # p = 6.639441e-06
print(kruskal.test(Malate.m.2 ~ pathologic_stage, data = df_filtered)$p.value) # p = 0.0001812289

# Multiple pairwise-comparison between groups
pairwise.t.test(df_filtered$Citrate.m.2, df_filtered$pathologic_stage, 
                     p.adjust.method = "BH") # significant between stage i & iii (FDR=0.0014), stage i & iv (FDR=0.0014)
pairwise.t.test(df_filtered$Succinate.m.2, df_filtered$pathologic_stage, 
                p.adjust.method = "BH") # significant between stage i & ii (FDR=0.04), i & iii (FDR=0.0011), stage i & iv (FDR=3e-05)
pairwise.t.test(df_filtered$Malate.m.2, df_filtered$pathologic_stage, 
                p.adjust.method = "BH")  # significant between stage i & iii (FDR=0.06686), stage i & iv (FDR=0.00015)

######## (3) Fig. 5c 
# Box plots of isotopologue levels between primary and metastatic samples in IMmotion151
data_dir = "/data1/resnike/xiea1/MetabolicModel/final_results/data/fig_5/"
df <- read.csv(paste0(data_dir,"IMmotion151_isotope_imputed_mean_met_flipped_clinical.csv"), row.names = 1)
df$PRIMARY_VS_METASTATIC <- factor(df$PRIMARY_VS_METASTATIC, levels = c("PRIMARY", "METASTATIC"))

# Citrate m+2
p <- wilcox.test(df$Citrate.m.2 ~ df$PRIMARY_VS_METASTATIC, exact = FALSE)$p.value 
ggplot(df, aes(x=PRIMARY_VS_METASTATIC, y=Citrate.m.2)) +
  geom_boxplot(aes(fill = PRIMARY_VS_METASTATIC), outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("Sample Status") + 
  ylab("Abundance") +
  labs(title = "Citrate m+2") +
  scale_fill_brewer(palette="Pastel1")
#geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_5c_citratem+2_metastasis.pdf")),width = 1.75, height = 2.25)
print(p) # p =  4.821298e-13

# Succinate m+2
p <- wilcox.test(df$Succinate.m.2 ~ df$PRIMARY_VS_METASTATIC, exact = FALSE)$p.value 
ggplot(df, aes(x=PRIMARY_VS_METASTATIC, y=Succinate.m.2)) +
  geom_boxplot(aes(fill = PRIMARY_VS_METASTATIC), outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("Sample Status") + 
  ylab("Abundance") +
  labs(title = "Succinate m+2") +
  scale_fill_brewer(palette="Pastel1")
#geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_5c_succinatem+2_metastasis.pdf")),width = 1.75, height = 2.25)
print(p) # p =  3.203881e-10

# Malate m+2
p <- wilcox.test(df$Malate.m.2 ~ df$PRIMARY_VS_METASTATIC, exact = FALSE)$p.value 
ggplot(df, aes(x=PRIMARY_VS_METASTATIC, y=Malate.m.2)) +
  geom_boxplot(aes(fill = PRIMARY_VS_METASTATIC), outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("Sample Status") + 
  ylab("Abundance") +
  labs(title = "Malate m+2") +
  scale_fill_brewer(palette="Pastel1")
#geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_5c_Malatem+2_metastasis.pdf")),width = 1.75, height = 2.25)
print(p) # p =  1.959809e-09

### cerise's ggplot theme for publication
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







