library(dplyr)

dir()
setwd("/home/gabriel.gomes/r_cluster")


#### 1.2 Conhecendo a base de dados/ Definindo o problema ####

filmes <- read.csv('movies.csv',stringsAsFactors = F)
View(filmes)

## lendo os dados pre-processados
filmes_transf <- read.csv2('movies_transf.csv')
View(filmes_transf)

#### 2.2 Preparando a base de dados####

## normalizando a base de dados
dados_normalizados <- scale(filmes_transf) ## execu??o com erro

## excluindo as colunas movieId e Titulo da base 
filmes_transf <- filmes_transf %>% 
  select(-movieId, -titulo)

## normalizando a base de dados
dados_normalizados <- scale(filmes_transf)
dados_normalizados <- data.frame(dados_normalizados)

#### 3. Kmeans no R ####
#### 3.1 Criando Cluster com R ####
set.seed(1987) #função pra gerar resultados iguais

## criando cluster para 3 agrupamentos
resultado_cluster <- kmeans(dados_normalizados,centers = 3)

## verificando resultado da função kmeans
resultado_cluster

## vetor com os clusters para cada registro
resultado_cluster$cluster

## dataframe com os centros para cada agrupamento para cada variavel(coluna)
resultado_cluster$centers

View(resultado_cluster$centers)

## vetor com as soma dos quadrados dentro do cluster. Espera-se que esse valor seja o mais baixo poss?vel para cada cluster, 
# porque se deseja ter homogeneidade dentro dos clusters
resultado_cluster$withinss

## Soma dos quadrados entre os clusters. Resulta a M?dia de distancia entre os centros de cada cluster,
# para se ter cluster heterogeneos, este valor deve ser o mais algo poss?vel.
resultado_cluster$betweenss  

## tamanho de cada cluster
resultado_cluster$size

#### 3.2 Plotando os clusters ####

## criando visualização dos cluster com o pacote 'cluster'
#install.packages("cluster")
library(cluster)
clusplot(x = dados_normalizados, resultado_cluster$cluster, color=TRUE, shade=TRUE)#, labels=3)#lines=3,


## criando visualização dos cluster com o pacote 'fpc'
#install.packages("fpc")
library(fpc)
plotcluster(x = dados_normalizados, resultado_cluster$cluster,ignorenum = T)


#### 3.3 Visualizando Generos nos Clusters(agrupamentos) ####

## criando gráficos com os centros
centros <- resultado_cluster$centers

#install.packages('reshape2')
library(reshape2)
centros_2 <- melt(centros) #,id.vars = 'genero')

#atribuindo novos nomes da coluna e convertendo coluna cluster para factor
colnames(centros_2) <- c('cluster','genero','centro')
centros_2$cluster   <- as.factor(centros_2$cluster)

# criando gráficos com a lib ggplot2, pra cada cluster
#install.packages('ggplot2')
library(ggplot2)
ggplot(data = centros_2 ) + 
  geom_bar(aes(x = genero,y = centro,fill = cluster),stat = 'identity') + 
  facet_grid(cluster ~ .)

#### 4. ESCOLHENDO O MELHOR NÚMERO K####

#### 4.1 Elbow ####
range_k        <- c(1:25) ## número de cluster que serão testados de 1 a 25 cluster
soma_quadrados <- 0 ## objeto que armazenara o resultado da soma dos `withinss`

set.seed(1987) # não altere para que seu resultados correspondam igual ao demosntrado na aula
## loop que executara o algoritmo kmeans para cada número de cluster do range_k
for (i in range_k){
  soma_quadrados[i] <- sum(kmeans(dados_normalizados,centers = i,nstart = 25)$withinss) 
} 
soma_quadrados

plot(range_k, soma_quadrados, type="b", xlab="Número de Clusters", ylab=" Soma dos Quadrados(grupos Within)")
axis(side=1,at=range_k, labels=range_k)

# Voce só poderá defini o valor abaixo depois de analisar o gráfico acima o ponto de 'cotovelo'
abline(v=5,col="red")


#### 4.2 Silhouette ####
media_silhouete <- c(0) ## criando vetor que ira armazenar a média dos valores de silhouette
range_k         <- c(2:20) ## valores de cluster que serão testados (deve ser maior ou igual a 2)

set.seed(1987) # não altere para que seu resultados correspondam igual ao demosntrado na aula
## loop para calcular a silhouette para cada numero de cluster (range_k)
for (i in range_k){
  print(i)
  clusters  <- kmeans(dados_normalizados,  centers = i) ## criando cluster
  Silhouete <- silhouette(clusters$cluster, dist(dados_normalizados)) ## calculando a silhouette
  media_silhouete[i]   <-  mean(Silhouete[,3]) ## calculando a media da silhouette
}
media_silhouete 

## criando gráfico com a média da silhoutte para cada número de cluster
plot(media_silhouete ,
     type = "b", 
     xlab = "Número de Cluster(k)",
     ylab = "Média Silhouettes")
axis(side=1,at=range_k, labels=range_k)

# Voce só poderá definir o valor abaixo depois de inspecionar no grafico acima
abline(v=12,col="red")


#### 4.3 Criando recomendação de filmes ####
set.seed(1987) #função pra gerar resultados iguais

## criando cluster para 10 agrupamentos
resultado_cluster <- kmeans(dados_normalizados,centers = 12)

## criando gráficos com os centros
centros <- resultado_cluster$centers

## Criando gráficos os cluster
centros_2           <- melt(centros)
colnames(centros_2) <- c('cluster','genero','centro')
centros_2$cluster   <- as.factor(centros_2$cluster)

# criando gráficos com a lib ggplot2, pra cada cluster
ggplot(data = centros_2 ) + 
  geom_bar(aes(x = genero,y = centro,fill = cluster),stat = 'identity') + 
  facet_grid(cluster ~ .)

## Atribuindo cluster para cada filme
filmes$cluster <- resultado_cluster$cluster
View(filmes)

## fazendo recomendação a partir de um filme
agrupamento <- filmes[filmes$title == 'Toy Story (1995)','cluster']
agrupamento

## selecinando 10 filmes dentro do cluster
filmes[filmes$cluster == agrupamento, 'title'] %>% 
  sample(10)

#### 5. CLUSTER HIERARQUICO ####
#### 5.1 Construindo um cluster hierarquico ####

set.seed(1987) # não altere para que seu resultados correspondam igual ao demosntrado na aula
matriz_dist <- dist(centros) #method = "euclidean"  # matrix de distância
matriz_dist

## criando objeto de cluster hierarquico 
clust_h <- hclust(matriz_dist) #, method = "complete")

## grafico com o cluster hierarquico
plot(clust_h)
plot(clust_h,hang = -1)


