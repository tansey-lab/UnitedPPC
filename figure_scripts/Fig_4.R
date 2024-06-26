# R scripts plotting Figure. 4 (How do genetic alterations interplay with metabolic adaptions in ccRCC?)
plots_dir = "/work/reznik/xiea1/MetabolicModel/final_results/figures/fig_4/"


######## (1) Fig. 4b
# A bar plot showing the number of significantly associated metabolites for ccRCC driver genes
# load all wilcoxon rank sum test results of Mutation(WT/MUT) ~ Metabolite for all 14 driver genes
data_dir = "/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/TCGA/downstream_analysis/wilcox_df_tcga/"
results_df <- matrix(nrow = 0, ncol = 1)  
results_df <- as.data.frame(results_df) 
colnames(results_df) <- c("significant_metabolites")
for (gene in list('VHL', 'PBRM1', 'SETD2', 'BAP1', 'MTOR', 'KDM5C', 'PIK3CA', 
                  'PIK3R1', 'PTEN', 'TP53', 'TSC1', 'TSC2', 'TCEB1', 'SDHB')){
  df <- read.csv(paste0(data_dir,"wilcox_",gene,"_62met.csv"), row.names = 1)
  sig_met_count <- nrow(df %>% filter(p_adj < 0.1))
  results_df[gene, ] <- sig_met_count 
}
# sort
results_df <- results_df[order(-results_df$significant_metabolites), , drop = FALSE]
desired_order <- factor(rownames(results_df), levels = unique(rownames(results_df)))

# generate a bar plot
ggplot(results_df, aes(x = rownames(results_df), y = significant_metabolites)) +
  geom_bar(stat = "identity", fill = "light blue") +
  theme_Publication() +
  labs(x = "Mutation", y = "Number of Metabolites with significant associations") +
  scale_x_discrete(limits = desired_order)
ggsave(file.path(paste0(plots_dir,"fig_4b_sig.pdf")),width = 5,height = 3) 

# A bar plot showing the fraction of mutants for ccRCC driver genes
df <- read.csv(paste0(data_dir,"mut_met_flipped.csv"), row.names = 1)[, 1:14]
results_df_2 <- matrix(nrow = 14, ncol = 0)  
results_df_2 <- as.data.frame(results_df_2) 
rownames(results_df_2) <- colnames(df)
results_df_2['mut_ratio'] = colSums(df)/nrow(df)*100

ggplot(results_df_2, aes(x = rownames(results_df_2), y = mut_ratio)) +
  geom_bar(stat = "identity", fill = "dark grey") +
  theme_Publication() +
  labs(x = "Mutation", y = "Mutation Frequency in TCGA samples (%)") +
  scale_x_discrete(limits = desired_order)
ggsave(file.path(paste0(plots_dir,"fig_4b_freq.pdf")),width = 5,height = 2) 


######## (2) Fig. 4c
# Pathway-based analysis of imputed metabolic changes in Mutant vs. WT samples in the TCGA KIRC cohort
# Define functions to calculate DA scores and plot figures
calculate_da <- function(pathways,data){
  da_scores <- as.data.frame(matrix(NA, nrow = length(pathways),ncol = 5))
  colnames(da_scores) <- c("Metabolite_Count","Up","Down","DA_Score","Pathway")
  da_scores$Metabolite_Count <- sapply(1:length(pathways),function(i){dim(data[data$pathway == pathways[i],])[1]})
  da_scores$Up <- sapply(1:length(pathways),function(i){dim(data[data$pathway == pathways[i] & data$mean.difference > 0 & data$Significant == "FDR < 0.1",])[1]}) 
  da_scores$Down <- sapply(1:length(pathways),function(i){dim(data[data$pathway == pathways[i] & data$mean.difference < 0 & data$Significant == "FDR < 0.1",])[1]}) 
  da_scores$DA_Score <- sapply(1:length(pathways), function(i){(da_scores$Up[i]-da_scores$Down[i])/da_scores$Metabolite_Count[i]})
  da_scores$Pathway <- pathways 
  da_scores$Pathway <- factor(da_scores$Pathway, levels = da_scores[order(da_scores$DA_Score),5])
  return(da_scores)
}

plot_da <- function(data,title){
  ggplot(data, aes(x=Pathway, y=DA_Score)) +
    geom_segment(aes(x=Pathway, xend=Pathway, y=0, yend=DA_Score), color=ifelse(data$DA_Score > 0,"red","blue")) +
    geom_point(color=ifelse(data$DA_Score > 0,"red","blue")) + geom_hline(yintercept = 0, col = "grey") + coord_flip() +
    theme_Publication() +
    xlab("") +
    ylab("Differential Abundance Score") +
    ggtitle(paste(title))
}
data_dir = "/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/TCGA/downstream_analysis/wilcox_df_tcga/"
# Load metabolic pathway annotations
metanno <- read.csv("/work/reznik/xiea1/MetabolicModel/data/metabolite_annotations.csv")
for (gene in list('VHL', 'PBRM1', 'SETD2', 'BAP1', 'TP53')){
  wilcox_df <- read.csv(paste0(data_dir,"wilcox_",gene,"_62met.csv"))
  colnames(wilcox_df)[1] <- 'metabolite'
  df <- wilcox_df %>% left_join(metanno, by="metabolite")
  df$Significant <- ifelse(df$p_adj < 0.1,"FDR < 0.1","Not Significant")
  # only keep pathways with 3+ sig altered metabolites 
  pathways <- names(which(sort(table(subset(df,Significant == "FDR < 0.1")[,'pathway']),decreasing = TRUE) >= 1)) 
  df_pathways <- df[df$pathway %in% pathways,]
  df_pathways$sub_pathway <- as.factor(df_pathways$pathway)
  pathways_da <- calculate_da(pathways,df_pathways)
  plot_da(pathways_da,gene)
  ggsave(file.path(paste0(plots_dir,"fig_4c_", gene, "_tcga.pdf")),width = 4, height = 4) 
}

# (3) Fig. 4d
# Box plots showing the levels of imputed glucose and lactate in BAP1 WT and MUT samples (TCGA)
data_dir = "/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/TCGA/downstream_analysis/wilcox_df_tcga/"
df <- read.csv(paste0(data_dir,"mut_met_flipped.csv"), row.names = 1)
df$BAP1 <- ifelse(df$BAP1== 0, 'WT', 'MUT')

# A box plot of imputed glucose levels (TCGA)
p <- t.test(df$glucose~df$BAP1)$p.value # p = 8.702785e-07
ggplot(df, aes(x=BAP1, y=glucose)) +
  geom_boxplot(aes(fill = BAP1),outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("BAP1 Status") + 
  ylab("Abundance") +
  labs(title = "Glucose") +
  scale_fill_brewer(palette="Pastel1")
  #geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_4d_glucose.pdf")),width = 1.75, height = 2.25) 
print(p) # p = 2.646601e-07

# A box plot of imputed citrate levels (TCGA)
p <- t.test(df$citrate~df$BAP1)$p.value
ggplot(df, aes(x=BAP1, y=citrate)) +
  geom_boxplot(aes(fill = BAP1),outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("BAP1 Status") + 
  ylab("Abundance") +
  labs(title = "Citrate") +
  scale_fill_brewer(palette="Pastel1")
  #geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_4d_citrate.pdf")),width = 1.75, height = 2.25)
print(p) # p = 0.02790222

# A box plot of imputed fumarate levels (TCGA)
p <- t.test(df$fumarate~df$BAP1)$p.value
ggplot(df, aes(x=BAP1, y=fumarate)) +
  geom_boxplot(aes(fill = BAP1),outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("BAP1 Status") + 
  ylab("Abundance") +
  labs(title = "Fumarate") +
  scale_fill_brewer(palette="Pastel1")
  #geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_4d_fumarate.pdf")),width = 1.75, height = 2.25) 
print(p) # p = 0.00715812

# A box plot of imputed malate levels (TCGA)
p <- t.test(df$malate~df$BAP1)$p.value
ggplot(df, aes(x=BAP1, y=malate)) +
  geom_boxplot(aes(fill = BAP1),outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("BAP1 Status") + 
  ylab("Abundance") +
  labs(title = "Malate") +
  scale_fill_brewer(palette="Pastel1")
  #geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_4d_malate.pdf")),width = 1.75, height = 2.25) 
print(p) # p = 0.007551505

######## (4) Fig. 4f
# Box plots showing the levels of imputed citrate m+2, succinate m+2, malate m+2 in BAP1 WT and MUT samples (TCGA KIRC)
data_dir <-'/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/KIRC_isotope_pyr/downstream_analysis/'
df <- read.csv(paste0(data_dir,"mut_isotope_kirc_flipped.csv"), row.names = 1)
df$BAP1 <- ifelse(df$BAP1== 0, 'WT', 'MUT')

# Citrate m+2 (KIRC_isotope_pyr)
p <- t.test(df$Citrate.m.2 ~ df$BAP1, exact = FALSE)$p.value 
ggplot(df, aes(x=BAP1, y=Citrate.m.2)) +
  geom_boxplot(aes(fill = BAP1), outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("BAP1 Status") + 
  ylab("Abundance") +
  labs(title = "Citrate m+2") +
  scale_fill_brewer(palette="Pastel1")
#geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_4f_citratem2_tcga.pdf")),width = 1.75, height = 2.25)
print(p) # p = 0.0002850083

# Succinate m+2 (KIRC_isotope_pyr)
p <- t.test(df$Succinate.m.2 ~ df$BAP1, exact = FALSE)$p.value 
ggplot(df, aes(x=BAP1, y=Succinate.m.2)) +
  geom_boxplot(aes(fill = BAP1), outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("BAP1 Status") + 
  ylab("Abundance") +
  labs(title = "Succinate m+2") +
  scale_fill_brewer(palette="Pastel1")
#geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_4f_succinatem2_tcga.pdf")),width = 1.75, height = 2.25)
print(p) # p =  0.003201255

# Malate m+2 (KIRC_isotope_pyr)
p <- t.test(df$Malate.m.2 ~ df$BAP1, exact = FALSE)$p.value 
ggplot(df, aes(x=BAP1, y=Malate.m.2)) +
  geom_boxplot(aes(fill = BAP1),outlier.size = 0.1) +
  geom_jitter(width = 0.2, size=0.1, alpha = 0.1) +
  theme_Publication() +
  xlab("BAP1 Status") + 
  ylab("Malate.m.2 Abundance") +
  labs(title = "Malate m+2") +
  scale_fill_brewer(palette="Pastel1")
#geom_text(aes(x = 2.5, y = 0.03, label = sprintf("Wilcoxon test,\n p = %.3f", p)))
ggsave(file.path(paste0(plots_dir,"fig_4f_malatem2_tcga.pdf")),width = 1.75, height = 2.25)
print(p) # p = 0.03040283

### ggplot theme for publication
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