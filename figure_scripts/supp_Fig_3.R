# R scripts plotting supplementary Figure. 3

#Load packages
library(ggpubr)
library(ggplot2)
library(tidyverse)
library(readxl)
library(msigdbr)

# define plot path
plots_dir = "/data1/reznike/xiea1/MetabolicModel/final_results/figures/supp_fig_3/"

###########(1) Fig. supp_3c TCA cycle signature v.s. true isotopologue (citrate m+2)
data_path="/data1/reznike/xiea1/"

 # 1.1 NSCLC dataset
# Read in RNA data
rna <- read.csv(paste0(data_path,"MetabolicModel/data/RNA_matched_NSCLC_G6/matched_tpm_NSCLC_G6_1.csv"))
rna <- dplyr::rename(rna, sample = X)

# Read in ground-truth NSCLC isotope data
isotope <- read.csv(paste0(data_path,"MetabolicModel/data/MET_matched_NSCLC_G6/matched_isotope_NSCLC_G6_1.csv"))
isotope <- dplyr::rename(isotope, sample = X)

# Load Reactome gene set names
reactome<- msigdbr(species = "Homo sapiens", category = "C2", subcategory = 'CP:REACTOME')
reactome_gene_sets <- split(reactome$gene_symbol, reactome$gs_name)

isotopologue_name <- 'CitG6m2'
gene_set_name <- "REACTOME_CITRIC_ACID_CYCLE_TCA_CYCLE"
signature <- reactome_gene_sets[[gene_set_name]]
rna$signature <- rowSums(rna[, names(rna) %in% signature])
df <- data.frame(isotopologue=(rank(isotope[,isotopologue_name],ties.method = 'min')-1)/nrow(isotope), signature=rna$signature) 
df %>% 
  ggplot(aes(x = isotopologue, y=signature))+
  geom_point(size=0.4,  color="#88CCEE")+
  geom_smooth(method = "lm", se = FALSE, linetype = "solid", linewidth=0.5, color = "black") +  # Add this line for the regression
  theme_Publication()+
  scale_color_manual()+
  scale_x_continuous()+
  scale_y_continuous()+
  labs(x = isotopologue_name,
       y = gene_set_name)+
  theme(aspect.ratio = 1) 
ggsave(file.path(plots_dir,"fig_3c_Citrate m+2_gene_NSCLC_G6_reatome.pdf"), device = "pdf", width = 2, height = 3)

rho <- cor.test(df$isotopologue, df$signature, method="spearman")[["estimate"]]
p <- cor.test(df$isotopologue, df$signature, method="spearman")[["p.value"]]
print(c(rho, p)) # reactome tca: rho=0.05907139, p=0.70937242 

# 1.2 RCC-glc dataset
# Read in RNA data
rna <- read.csv(paste0(data_path,"MetabolicModel/data/RNA_matched_MITO/matched_tpm_MITO1.csv"))
rna <- dplyr::rename(rna, sample = X)

# Read in ground-truth MITO1 isotope data
isotope <- read.csv(paste0(data_path,"MetabolicModel/data/MET_matched_MITO/matched_isotope_MITO1.csv"))
isotope <- dplyr::rename(isotope, sample = X)

# Read in predicted MITO1 isotope data
predicted_isotope <- read.csv(paste0(data_path,"MetabolicModel/results_RNA_MITO/MITO1/actual_vs_predicted_ranks.csv"))
lactate <- predicted_isotope[predicted_isotope$feature=='Lactate m+3',]

# Load Reactome gene set names
reactome<- msigdbr(species = "Homo sapiens", category = "C2", subcategory = 'CP:REACTOME')
reactome_gene_sets <- split(reactome$gene_symbol, reactome$gs_name)

# RCC
isotopologue_name <- 'Citrate.m.2'
gene_set_name <- "REACTOME_CITRIC_ACID_CYCLE_TCA_CYCLE"
signature <- reactome_gene_sets[[gene_set_name]]

rna$signature <- rowSums(rna[, names(rna) %in% signature])
df <- data.frame(isotopologue=(rank(isotope[,isotopologue_name],ties.method = 'min')-1)/nrow(isotope), signature=rna$signature) 
df %>% 
  ggplot(aes(x = isotopologue, y=signature))+
  geom_point(size=0.4,  color="#DDCC77")+
  geom_smooth(method = "lm", se = FALSE, linetype = "solid", linewidth=0.5, color = "black") +  # Add this line for the regression
  theme_Publication()+
  scale_color_manual()+
  scale_x_continuous()+
  scale_y_continuous()+
  labs(x = isotopologue_name,
       y = gene_set_name)+
  theme(aspect.ratio = 1) 
ggsave(file.path(plots_dir,"supp_fig_3c_Citrate m+2_gene_MITO_reactome.pdf"), device = "pdf", width = 2, height = 3)

rho <- cor.test(df$isotopologue, df$signature, method="spearman")[["estimate"]]
p <- cor.test(df$isotopologue, df$signature, method="spearman")[["p.value"]]
print(c(rho, p)) # reactome TCA: rho=0.2496033, p=0.1307133 


# (2) Gene signature v.s. true isotopologue (lactate m+3)
# Load hallmark gene set names
# get a list of gene sets to check
hallmark<- msigdbr(species = "Homo sapiens", category = "H")
hallmark_gene_sets <- split(hallmark$gene_symbol, hallmark$gs_name)

# glycolysis vs lactate m+3
isotopologue_name <- 'Lactate.m.3'
gene_set_name <- "HALLMARK_GLYCOLYSIS"
signature <- hallmark_gene_sets[[gene_set_name]]
rna$signature <- rowSums(rna[, names(rna) %in% signature])
df <- data.frame(isotopologue=(rank(isotope[,isotopologue_name],ties.method = 'min')-1)/nrow(isotope), signature=rna$signature) 
df %>% 
  ggplot(aes(x = isotopologue, y=signature))+
  geom_point(size=0.4,  color="#DDCC77")+
  geom_smooth(method = "lm", se = FALSE, linetype = "solid", linewidth=0.5, color = "black") +  # Add this line for the regression
  theme_Publication()+
  scale_color_manual()+
  scale_x_continuous()+
  scale_y_continuous()+
  labs(x = isotopologue_name,
       y = gene_set_name)+
  theme(aspect.ratio = 1) 
ggsave(file.path(plots_dir,"supp_fig_3c_Lactate m+3_gene_MITO.pdf"), device = "pdf", width = 2, height = 3)

rho <- cor.test(df$isotopologue, df$signature, method="spearman")[["estimate"]]
p <- cor.test(df$isotopologue, df$signature, method="spearman")[["p.value"]]
print(c(rho, p)) # rho=0.05153737, p=0.75797388 

# true vs predicted lactae m+3
df <- data.frame(true_flipped=(nrow(lactate)-lactate$actual_rank)/nrow(lactate), predicted_flipped=(nrow(lactate)-lactate$predicted_rerank)/nrow(lactate)) 
df %>% 
  ggplot(aes(x = true_flipped, y=predicted_flipped))+
  geom_point(size=0.4,  color="#DDCC77")+
  geom_smooth(method = "lm", se = FALSE, linetype = "solid", linewidth=0.5, color = "black") +  # Add this line for the regression
  theme_Publication()+
  scale_color_manual()+
  scale_x_continuous()+
  scale_y_continuous()+
  labs(x = 'True rank',
       y = 'Predicted rank',
       title='Lactae m+3')+
  theme(aspect.ratio = 1) 
ggsave(file.path(plots_dir,"supp_fig_3c_Lactate m+3_MITO.pdf"), device = "pdf", width = 2, height = 3)

rho <- cor.test(df$true_flipped, df$predicted_flipped, method="spearman")[["estimate"]]
p <- cor.test(df$true_flipped, df$predicted_flipped, method="spearman")[["p.value"]]
print(c(rho, p)) # rho=0.430134588,p= 0.007484466 