library(readr)
library(dplyr)
library(cluster)
library(fpc)
library(reshape2)

##Refazendo com uma maoir quantidade de clusters

filmes <- read_csv("C:/Users/LucasZan/Desktop/Alura/1527-cluster_r-data/movies.csv")

df <- read_delim("C:/Users/LucasZan/Desktop/Alura/1527-cluster_r-data/movies_transf.csv", 
                 ";", escape_double = FALSE, trim_ws = TRUE)


filmes_tranformados <- df %>% 
  select(-movieId,-titulo)

dados_normalizados <- data.frame( scale(filmes_tranformados) )


set.seed(123)

resultado_cluster_2 <- kmeans(dados_normalizados, centers = 7)

resultado_cluster_2$size
resultado_cluster_2$withinss
#resultado_cluster_2$centers


clusplot(x = dados_normalizados, resultado_cluster_2$cluster,
         color=TRUE, shade=TRUE)

plotcluster(x= filmes_tranformados, resultado_cluster_2$cluster, ignorenum = T)


ggplot(data = centros_2) +
  geom_bar(aes(x= genero, y= centro, fill = cluster), stat = 'identity') +
  facet_grid(cluster ~ .)


plot(range_k, soma_dos_quadrados, type = 'b',
     xlab = 'Numero de clusters',
     ylab = 'Soma dos Quadrados')
axis(side = 1, at = range_k, labels = range_k)