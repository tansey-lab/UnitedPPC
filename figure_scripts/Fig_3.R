# R scripts plotting Figure. 3

#Load packages
library(ggpubr)
library(ggplot2)
library(tidyverse)
library(readxl)
library(msigdbr)

# define plot path
plots_dir = "/data1/reznike/xiea1/MetabolicModel/final_results/figures/fig_3/"

########## (1) Fig. 3c
# Scatterplots comparing measured and predicted isotopologue ranks in MITO and NSCLC_G6
predicted_vs_actual_scatter <- function(scatter.dat,
         met,
         palette = c("lower","upper")){
  if (palette == "lower"){
    pal <- batch.palette.lower
  } else if (palette == "upper"){
    pal <- batch.palette
  }
  
  scatter.dat %>% 
    filter(metabolite == met) %>% 
    ggplot(aes(x = true.rank,y=predicted.rank,color = batch))+
    geom_point(size=0.4)+
    geom_smooth(method = "lm", se = FALSE, linetype = "solid", linewidth=0.5, color = "black") +  # Add this line for the regression
    theme_Publication()+
    scale_color_manual(values = pal)+
    scale_x_continuous()+
    scale_y_continuous()+
    labs(x = "True rank",
         y = "Predicted rank",
         title = str_to_title(met))+
    theme(aspect.ratio = 1) 
}
## SCATTERPLOTS OF ACTUAL VS PREDICTED RANKS FOR SELECTED METABOLITES ----
##### NSCLC_G
results_dir <- "/data1/reznike/xiea1/MetabolicModel/results_RNA_isotope"
aggregate_act.pred <- map_dfr(list.dirs(results_dir,recursive = F) %>% 
                                grep(invert = T, value = T,pattern ="logs"),
                              function(dir){
                                bname <- gsub(dir,pattern= "/data1/reznike/xiea1/MetabolicModel/results_RNA_isotope/",replacement = "")
                                ave_actual_vs_predicted_ranks <- read.csv(paste0(dir,"/actual_vs_predicted_ranks_flipped.csv"))
                                ave_actual_vs_predicted_ranks <- ave_actual_vs_predicted_ranks %>% mutate(batch = bname) %>% rename(metabolite = feature,
                                                                                                                                    true.rank = flipped_actual_rank,
                                                                                                                                    predicted.rank = flipped_predicted_rerank,
                                                                                                                                    sample.index = sample_index)
                              })

# set batch palette 
aggregate_act.pred <- aggregate_act.pred[aggregate_act.pred$batch =='NSCLC_G6',]
batch.palette <- c("NSCLC_G6" = "#88CCEE") # "MITO" = "#DDCC77", "NSCLC_G6" = "#88CCEE"
# lowercase version of batch palette
batch.palette.lower <- batch.palette
names(batch.palette.lower) <- tolower(names(batch.palette))

# mutate isotopologue names in NSCLC_G6 because the naming ways are different
aggregate_act.pred <- aggregate_act.pred %>% mutate (metabolite)
aggregate_act.pred$metabolite <- sub("CitG6m2", "Citrate m+2", aggregate_act.pred$metabolite)


#Plot scatter plots
# Citrate m+2 (average rho: 0.445)
aggregate_act.pred %>% predicted_vs_actual_scatter(met = "Citrate m+2", palette = "upper")
ggsave(file.path(plots_dir,"fig_3c_Citrate m+2_NSCLC_G6.pdf"), device = "pdf", width = 2, height = 3)

##### MITO
results_dir <- "/data1/reznike/xiea1/MetabolicModel/results_RNA_isotope"
aggregate_act.pred <- map_dfr(list.dirs(results_dir,recursive = F) %>% 
                                grep(invert = T, value = T,pattern ="logs"),
                              function(dir){
                                bname <- gsub(dir,pattern= "/work/reznik/xiea1/MetabolicModel/results_RNA_isotope/",replacement = "")
                                ave_actual_vs_predicted_ranks <- read.csv(paste0(dir,"/actual_vs_predicted_ranks_flipped.csv"))
                                ave_actual_vs_predicted_ranks <- ave_actual_vs_predicted_ranks %>% mutate(batch = bname) %>% rename(metabolite = feature,
                                                                                                                                    true.rank = flipped_actual_rank,
                                                                                                                                    predicted.rank = flipped_predicted_rerank,
                                                                                                                                    sample.index = sample_index)
                              })

# set batch palette 
aggregate_act.pred <- aggregate_act.pred[aggregate_act.pred$batch =='MITO',]
batch.palette <- c("MITO" = "#DDCC77") # "MITO" = "#DDCC77", "NSCLC_G6" = "#88CCEE"
# lowercase version of batch palette
batch.palette.lower <- batch.palette
names(batch.palette.lower) <- tolower(names(batch.palette))

#Plot scatter plots
# Citrate m+2 (average rho: 0.395)
aggregate_act.pred %>% predicted_vs_actual_scatter(met = "Citrate m+2", palette = "upper")
ggsave(file.path(plots_dir,"fig_3c_Citrate m+2_MITO.pdf"), device = "pdf", width = 2, height = 3)

###########(1.5) Fig. 3c Gene signature v.s. true isotopologue
data_path="/data1/reznike/xiea1/"
plots_dir <- paste0(data_path, "MetabolicModel/final_results/figures/fig_3/")
########### RCC-glc dataset
# Read in RNA data
rna <- read.csv(paste0(data_path,"MetabolicModel/data/RNA_matched_MITO/matched_tpm_MITO1.csv"))
rna <- dplyr::rename(rna, sample = X)

# Read in ground-truth MITO1 isotope data
isotope <- read.csv(paste0(data_path,"MetabolicModel/data/MET_matched_MITO/matched_isotope_MITO1.csv"))
isotope <- dplyr::rename(isotope, sample = X)

# Load hallmark gene set names
# get a list of gene sets to check
hallmark<- msigdbr(species = "Homo sapiens", category = "H")
hallmark_gene_sets <- split(hallmark$gene_symbol, hallmark$gs_name)

# oxidative phosphorylation vs citrate m+2
isotopologue_name <- 'Citrate.m.2'
gene_set_name <- "HALLMARK_OXIDATIVE_PHOSPHORYLATION"
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
ggsave(file.path(plots_dir,"fig_3c_Citrate m+2_gene_MITO_reactome.pdf"), device = "pdf", width = 2, height = 3)

rho <- cor.test(df$isotopologue, df$signature, method="spearman")[["estimate"]]
p <- cor.test(df$isotopologue, df$signature, method="spearman")[["p.value"]]
print(c(rho, p)) # oxphos:rho=0.2195109,p= 0.1854462; reactome TCA: rho=0.2496033, p=0.1307133 

########### NSCLC_G6 dataset
# Read in RNA data
rna <- read.csv(paste0(data_path,"MetabolicModel/data/RNA_matched_NSCLC_G6/matched_tpm_NSCLC_G6_1.csv"))
rna <- dplyr::rename(rna, sample = X)

# Read in ground-truth MITO1 isotope data
isotope <- read.csv(paste0(data_path,"MetabolicModel/data/MET_matched_NSCLC_G6/matched_isotope_NSCLC_G6_1.csv"))
isotope <- dplyr::rename(isotope, sample = X)

# Load hallmark gene set names
# get a list of gene sets to check
hallmark<- msigdbr(species = "Homo sapiens", category = "H")
hallmark_gene_sets <- split(hallmark$gene_symbol, hallmark$gs_name)

# oxidative phosphorylation vs citrate m+2
isotopologue_name <- 'CitG6m2'
gene_set_name <- "HALLMARK_OXIDATIVE_PHOSPHORYLATION"
signature <- hallmark_gene_sets[[gene_set_name]]
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
print(c(rho, p)) # oxphos: rho=0.1639251 p=0.2984673; reactome tca: rho=0.05907139, p=0.70937242 



########## (2) Fig. 3e, g
# Umaps of trained MITO-KIPAN latent sample embedding W
generate_sample_umap <- function(W.path, by, seed = 42){
    W.mat <- read.csv(W.path) %>% 
      tibble::column_to_rownames(var="X") %>% 
      as.matrix()
    
    set.seed(seed)
    w.umap <- W.mat %>% 
      umap()
  
    w.umap.anno <- w.umap$layout %>% 
      as.data.frame() %>% 
      tibble::rownames_to_column(var=by) %>% 
      as_tibble() %>%
      # add sample type annotation
      left_join(sample.map)
  }

results_dir =  "/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/KIPAN/"
sample.map.path <-paste0(results_dir,"embeddings/sample_type_MITO_KIPAN_mismatched_dropped.csv")
sample.map <- read.csv(sample.map.path) %>% rename(sample.id = X)
W.path <- paste0(results_dir,"embeddings/W_loc_mismatched_dropped.csv")
batch_size = 965 # 965 correctly matched samples, 68 MITO samples (after dropping mismatched samples)

W_umap <- generate_sample_umap(W.path, by='sample.id')
#  Fig. 3e
# Color the Umap by batch (KIPAN/MITO)
batch.palette <- c("KIPAN" = "#4794D0",
                   "MITO" = "#DC3627")
W_umap %>% 
  ggplot(aes(x=V1,y=V2,color=batch))+
  geom_point(size=0.001)+
  theme_classic()+
  theme(axis.text = element_blank(),
        axis.ticks = element_blank())+
  #scale_color_brewer(palette = "Set1")+
  scale_color_manual(values = batch.palette)+
  labs(x="UMAP1",
       y="UMAP2")+
  theme(aspect.ratio = 1)
ggsave(file.path(plots_dir,"fig_3e_W_umap_kipan_mito.pdf"),width = 4,height = 2)

#  Fig. 3g
# Color the Umap by sample type (normal/ccRCC/pRCC/ChRCC)
W_umap[1:batch_size,] %>% 
  ggplot(aes(x=V1,y=V2,color=histological_type))+
  geom_point(size=0.1)+
  theme_classic()+
  theme(axis.text = element_blank(),
        axis.ticks = element_blank())+
  scale_color_brewer(palette = "Set2")+
  #scale_color_manual(values = batch.palette)+
  labs(x="UMAP1",
       y="UMAP2")+
  theme(aspect.ratio = 1)
ggsave(file.path(plots_dir,"fig_3g_W_umap_kipan_all.pdf"),width = 5,height = 2)

########## (3) Fig. 3f
# Scatter plots comparing differences of metabolite abundances between subtypes of kidney cancer in the measured MITO and imputed TCGA KIRC data
# Validate that imputed isotope data recapitulates human biology of kidney cancer
# chromophobe vs ccRCC differential analysis
results_dir =  "/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/KIPAN/"
results_path =paste0(results_dir,"/wilcox_cancer_subtype_mito_kipan_flipped.csv")
wilcox_df <- read.csv(results_path)

wilcox_df <- dplyr::rename(wilcox_df,metabolite = X)
wilcox_df <- wilcox_df %>% 
  mutate(sig.status = case_when(
    p_chromo>= 0.1 | p_adj_chromo_kipan >= 0.1 ~ "ns",
    mean.difference_chromo > 0 & mean.difference_chromo_kipan > 0 ~ "significant in both, consistent",
    mean.difference_chromo < 0 & mean.difference_chromo_kipan < 0 ~ "significant in both, consistent",
    T ~ "significant in both, inconsistent"))

count <- wilcox_df %>% count(sig.status)
cor = cor.test(wilcox_df$mean.difference_chromo, wilcox_df$mean.difference_chromo_kipan, method = "spearman")
cor_p = format(cor[["p.value"]], scientific = TRUE) # < 0.0001
cor_rho = cor[["estimate"]][["rho"]] # 0.8478733, pvalue=4.17375e-15

label_1 = c("Lactate m+3", "Alanine m+3", "Malate m+3", "Aspartate m+3")

wilcox_df %>% 
  ggplot(aes(x=mean.difference_chromo,
             y=mean.difference_chromo_kipan,
             color=sig.status))+
  geom_point()+
  geom_label_repel(data=. %>%
                     filter(metabolite %in% label_1),
                   aes(label = metabolite),
                   color="black",
                   label.size=0,
                   fill=NA,
                   min.segment.length = 0, 
                   max.overlaps = Inf)+
  scale_color_manual(values = c("ns" = "grey",
                                "significant in both, consistent" = "#332288",
                                "significant in both, inconsistent" = "red"))+
  scale_x_continuous(breaks = c(-0.5,0,0.5),limits = c(-0.5,0.5))+
  scale_y_continuous(breaks = c(-0.5,0,0.5),limits = c(-0.5,0.5))+
  theme(aspect.ratio = 1)+
  theme_Publication()+
  labs(x="MITO Chromophobe/ccRCC rank difference (measured)",
       y="KIPAN Chromophobe/ccRCC rank difference (imputed)",
       title = "MITO vs KIPAN") 
  #geom_text(aes(x = 0.35, y = -0.5, label = sprintf("Spearman's rho = %.*f,\n p = %s", 3, cor_rho, cor_p)))
ggsave(file.path(plots_dir,"fig_3f_mito_kipan_chromo_mean_difference.pdf"),width = 4,height = 4)

####### papillary vs ccRCC
results_path =paste0(results_dir,"/wilcox_cancer_subtype_mito_kipan_flipped.csv")
wilcox_df <- read.csv(results_path)

wilcox_df <- dplyr::rename(wilcox_df,metabolite = X)
wilcox_df <- wilcox_df %>% 
  mutate(sig.status = case_when(
    p_papi>= 0.1 | p_adj_papi_kipan >= 0.1 ~ "ns",
    mean.difference_papi > 0 & mean.difference_papi_kipan > 0 ~ "significant in both, consistent",
    mean.difference_papi < 0 & mean.difference_papi_kipan < 0 ~ "significant in both, consistent",
    T ~ "significant in both, inconsistent"))

count <- wilcox_df %>% count(sig.status)
cor = cor.test(wilcox_df$mean.difference_papi, wilcox_df$mean.difference_papi_kipan, method = "spearman")
cor_p = format(cor[["p.value"]], scientific = TRUE) # < 0.0001
cor_rho = cor[["estimate"]][["rho"]] # 0.7908597, pvalue=5.03842e-12

label_2 = c("Citrate m+2", "Succinate m+2", "Malate m+2", "Glutamate m+2", "Glutamine m+2", "Aspartate m+2")

wilcox_df %>% 
  ggplot(aes(x=mean.difference_papi,
             y=mean.difference_papi_kipan,
             color=sig.status))+
  geom_point()+
  geom_label_repel(data=. %>%
                     filter(metabolite %in% label_2),
                   aes(label = metabolite),
                   color="black",
                   fill=NA,
                   min.segment.length = 0, 
                   max.overlaps = Inf)+
  scale_color_manual(values = c("ns" = "grey",
                                "significant in both, consistent" = "#332288",
                                "significant in both, inconsistent" = "red"))+
  theme(aspect.ratio = 1)+
  theme_Publication()+
  labs(x="MITO Papillary/ccRCC rank difference (measured)",
       y="KIPAN Papillary/ccRCC rank difference (imputed)",
       title = "MITO vs KIPAN") 
  #geom_text(aes(x = 0.25, y = -0.5, label = sprintf("Spearman's rho = %.*f,\n p = %s", 3, cor_rho, cor_p)))
ggsave(file.path(plots_dir,"fig_3f_mito_kipan_papi_mean_difference.pdf"),width = 4,height = 4)

### function to get exact p values for spearman's correlation
spearman_correlation <- function(x, y) {
  n <- length(x)
  ranked_x <- rank(x)
  ranked_y <- rank(y)
  
  # Spearman's correlation formula
  1 - (6 * sum((ranked_x - ranked_y)^2)) / (n * (n^2 - 1))
}
# Calculating Spearman's correlation coefficient
correlation_coefficient <- spearman_correlation(x, y)
print(correlation_coefficient)
# Calculating p-value for Spearman's correlation
n <- length(x)
t_stat <- correlation_coefficient * sqrt((n - 2) / (1 - correlation_coefficient^2))
p_value <- 2 * pt(-abs(t_stat), df = n - 2)
print(p_value)


########## (3) Fig. 3g
# dot and line plots showing predicted lactate m+3/glc m+6 and citrate m+2/glc m+6 in ETC complex 1 mutant and WT samples in TCAG KICH cohort
results_dir <- '/data1/reznike/xiea1/MetabolicModel/results_RNA_imputation/KICH_isotope_pyr'
imputed_mean_met <- read.csv(file.path(results_dir, 'KICH_isotope_imputed_mean_met_flipped_glcm6_subset.csv'))
imputed_std_met <- read.csv(file.path(results_dir, 'KICH_isotope_imputed_se_met_glcm6_subset.csv'))


mutation <- read_excel('/data1/reznike/xiea1/MetabolicModel/data/TCGA_downstream/ChRCC_mtdna_mutation.xlsx',
                       sheet = 2, skip = 3)
# Remove the last two rows
mutation <- mutation %>% slice(1: (n() - 2))

# Define cx1_genes
cx1_genes <- c('MT-ND1', 'MT-ND2', 'MT-ND3', 'MT-ND4', 'MT-ND5', 'MT-ND6')
cx1_indices <- mutation %>% 
  filter(Gene %in% cx1_genes) %>% 
  pull(case) %>% 
  unique()
# Get unique cases for cx1_genes where mutation type is indel
cx1_indel_indices <- mutation %>% 
  filter(Gene %in% cx1_genes, `SNV/indel` == 'indel') %>% 
  pull(case) %>% 
  unique()
# Get unique cases for cx1_genes where mutation type is SNV
cx1_snv_indices <- mutation %>% 
  filter(Gene %in% cx1_genes, `SNV/indel` == 'SNV') %>% 
  pull(case) %>% 
  unique()

indel_indices <- cx1_indel_indices
snv_indices <- cx1_snv_indices
mut_indices <- cx1_indices


metabolite_list <- c('Lactate.m.3', 'Citrate.m.2')

# ---------------------- box plot showing all 1000 draws----------------------
for (met in metabolite_list) {
  # Sort the samples based on the mean value of the metabolite
  df_met <- imputed_mean_met %>%
    select(c(met,"TCGA.patient.code"))%>%
    arrange(desc(!!sym(met))) 
  met_std <- paste0(met,'_std')
  df_std <- imputed_std_met %>%
    select(c(met,"TCGA.patient.code")) %>%
    rename(!!met_std := !!sym(met))

  
  df <- df_met %>%
    left_join(df_std, by = "TCGA.patient.code")
  
  # Create the plot
  p <- ggplot(df, aes(x = reorder(rownames(df), desc(!!sym(met))), y = !!sym(met))) +
    geom_point(aes(color = case_when(
      TCGA.patient.code %in% indel_indices ~ 'Complex 1 indel',
      TCGA.patient.code %in% snv_indices ~ 'Complex 1 SNV',
      TRUE ~ 'WT'
    )), size = 3) +
    geom_errorbar(aes(ymin = !!sym(met) - df[[met_std]], ymax = !!sym(met) + df[[met_std]], width = 0.2), color = 'darkgrey') +
    scale_color_manual(values = c('Complex 1 indel' = 'salmon', 'Complex 1 SNV' = 'palegreen', 'WT' = 'skyblue')) +
    labs(x = 'Samples', y = paste(met, 'abundance'), title = paste('Predicted', met, 'abundance of KICH'), color = 'Sample Type') +
    theme_Publication()
  
  met <- gsub("\\.", "_", met)
  # Save the plot
  ggsave(filename = paste0(plots_dir, 'fig_3g_', met, '_glcm6_cx1.pdf'), plot = p, width = 10, height = 5, units = 'in')
}



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


