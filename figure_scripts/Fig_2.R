# R scripts plotting Figure. 2

#Load packages
library(ggpubr)
library(ggplot2)
library(tidyverse)

# define plot path
plots_dir = "/data1/reznike/xiea1/MetabolicModel/final_results/figures/fig_2/"
metanno <- read.csv("/data1/reznike/xiea1/MetabolicModel/data/metabolite_annotations.csv")

########## (1) Fig. 2c
# Violin plots of spearman's rho of naive Lasso, MIRTH, UnitedMet -- Benchmarking
data_path = "/data1/reznike/xiea1/MetabolicModel/results_RNA_ccRCC/final_benchmark_long_table_final.csv"
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
ggsave(file.path(paste0(plots_dir,"fig_2b_benchmark_rho_violin.pdf")),width = 8, height = 4) 

# box plot
ggplot(df, aes(x = method, y=rho, fill=dataset)) +
  geom_boxplot(fatten=2) +
  facet_wrap(~dataset, nrow=1) + 
  theme_Publication() +
  xlab("Dataset") + 
  ylab("Spearman rho") +
  scale_fill_brewer(palette="Pastel1") 
ggsave(file.path(paste0(plots_dir,"fig_2b_benchmark_rho_box.pdf")),width = 8, height = 4) 


########## (3) Fig. 2d
# Scatterplots comparing measured and predicted metabolite ranks in CPTAC, CPTAC_val, RC18, RC20
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
results_dir <- "/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC"
aggregate_act.pred <- map_dfr(list.dirs(results_dir,recursive = F) %>% 
                                grep(invert = T, value = T,pattern ="logs"),
                              function(dir){
                                bname <- gsub(dir,pattern= "/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/",replacement = "")
                                ave_actual_vs_predicted_ranks <- read.csv(paste0(dir,"/actual_vs_predicted_ranks_flipped.csv"))
                                ave_actual_vs_predicted_ranks <- ave_actual_vs_predicted_ranks %>% mutate(batch = bname) %>% rename(metabolite = feature,
                                                                                                                                    true.rank = flipped_actual_rank,
                                                                                                                                    predicted.rank = flipped_predicted_rerank,
                                                                                                                                    sample.index = sample_index)
                              })

# set batch palette 
batch.palette <- c("CPTAC" = "#DDCC77",
                   "CPTAC_val" = "#AA4499",
                   "RC18" = "#88CCEE",
                   "RC20" = "#44AA99")
# lowercase version of batch palette
batch.palette.lower <- batch.palette
names(batch.palette.lower) <- tolower(names(batch.palette))

#Plot scatter plots
# kynurenine (average rho: 0.52,0.60,0.73, 0.70)
aggregate_act.pred %>% predicted_vs_actual_scatter(met = "kynurenine", palette = "upper")
ggsave(file.path(plots_dir,"fig_2d_kynurenine.pdf"), device = "pdf", width = 3, height = 3)

# N-acetylneuraminate (average rho: 0.63,0.66,0.55, 0.71)
aggregate_act.pred %>% predicted_vs_actual_scatter(met = "N-acetylneuraminate", palette = "upper")
ggsave(file.path(plots_dir,"fig_2d_N-acetylneuraminate.pdf"), device = "pdf", width = 3, height = 3)



########## (4) Fig. 2e
# BARPLOT OF REPRODUCIBLY WELL-PREDICTED METABOLITES
data_path = "/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/rho_sig_in_all_datasets_June_8.csv"
df_rho <- read.csv(data_path)
# convert "reproducibly_well_predicted" column to the TRUE or FALSE boolean values in R
df_rho$reproducibly_well_predicted <- as.logical(df_rho$reproducibly_well_predicted)
# convert the column name of 'feature' to 'metabolite'
colnames(df_rho)[1] <- 'metabolite'

n.inset = 200
reprod.bar <- df_rho %>% 
  ggplot(aes(x = fct_reorder(metabolite, average_rho,.desc = T),
             y = average_rho,
             fill=reproducibly_well_predicted)) +
  # bar
  geom_bar(stat = "identity", width = 1) +
  # Dotted line near rho == 0
  geom_vline(xintercept = df_rho %>% 
               slice_min(abs(average_rho)) %>% 
               pull(metabolite) %>% 
               head(1),
             linetype = "dotted")+
  scale_y_continuous(breaks = c(-0.25,0,0.25,0.5,0.75,1.0)) +
  coord_cartesian(ylim=c(-0.3, 1)) +
  labs(y="Average"~rho,
       x="Metabolite") +
  theme_classic() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) +
  scale_fill_manual(values=c("TRUE"="steelblue","FALSE"="grey80"))

reprod.inset <-reprod.bar+ 
  coord_cartesian(ylim=c(0.4,1),xlim=c(0,n.inset))+
  geom_text(data=. %>% 
              slice_max(order_by = average_rho, n=n.inset) %>% 
              filter(reproducibly_well_predicted),
            aes(label=metabolite, y=average_rho+0.01),
            color="black",
            angle=90,
            hjust=0,size=2)+
  theme(legend.position = "none",
        axis.text = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank())

ggdraw() +
  draw_plot(reprod.bar + geom_rect(xmin=0,xmax=n.inset,ymin=0.4,ymax=0.85,fill="NA",color="black")) +
  draw_plot(reprod.inset, x = 0.3, y = .4, width = 0.5, height = 0.6)

ggsave(file.path(plots_dir,"fig_2c_reprod_well_predicted_bar.pdf"),width = 8,height = 6)   

## CLASSES OF REPRODUCIBLY WELL-PREDICTED METABOLITES ----
# super pathway/metabolite class palette
super.pathway.palette <- c("Amino acid" = "#E69F00",
                           "Carbohydrate"= "#56B4E9",
                           "Cofactors and vitamins" ="#009E73",
                           "Energy" = "#F0E442",
                           "Lipid"= "#0072B2", 
                           "Nucleotide"= "#D55E00", 
                           "Peptide"= "#CC79A7",
                           "Xenobiotics" =  "#999999")
# grey color for missing values
na.grey <- "#E3E3E3"


reprod.well.predict.res <- df_rho %>% 
  left_join(metanno) %>% 
  filter(reproducibly_well_predicted) %>% 
  mutate(id = "reproducibly well-predicted") %>% 
  bind_rows(df_rho %>% 
              left_join(metanno) %>% 
              mutate(id = "all available")) %>% 
  select(metabolite,class,id) 
# Although I added these metabolites to metanno, unidentified problem existed. I have to do this manually here.
reprod.well.predict.res[reprod.well.predict.res$metabolite=='N,N-Diethylethanolamine','class'] = 'Lipid'
reprod.well.predict.res[reprod.well.predict.res$metabolite=='4-androsten-3beta,17beta-diol disulfate (1)*',
                        'class'] = 'Lipid'
reprod.well.predict.res[804,'class'] = 'Amino Acid' # tryptophan betaine
# both 'Amino Acid' and 'Amino acid' exit, unify to 'Amino acid'
reprod.well.predict.res[reprod.well.predict.res$class=='Amino Acid','class'] = 'Amino acid'

reprod.well.predict.res %>% 
  ggplot(aes(y=id,fill=class))+
  geom_bar(position = "fill")+
  theme_classic()+
  scale_fill_manual(values = super.pathway.palette,
                    na.value = na.grey)

ggsave(file.path(plots_dir,"fig_2c_reprod_well_predicted_class.pdf"), width=5, height=3)

## PATHWAYS OF REPRODUCIBLY WELL-PREDICTED METABOLITES ----
calc_reprod_well_predicted(single.set.by.metab) %>% 
  filter(reprod.well.predicted) %>% 
  left_join(metanno %>% mutate(metabolite = tolower(metabolite))) %>%
  group_by(super.pathway) %>% 
  summarise(n.in.pathway = n()) %>% 
  write_csv(file.path(plot.dir,"f2e_single_set_classes_well_predicted.csv"))

reprod_well_predicted_class(calc_reprod_well_predicted(single.set.by.metab))
ggsave(file.path(plot.dir,"f2e_sset_well_predict_metab_class.pdf"), width = 5, height = 3)



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



