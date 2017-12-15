rm(list=ls())
library('mlbench')
library('e1071')
library('caret')
library('clusterSim')
library('rgl')
library('readr')

t <- Sys.time()

#Algoritmo Diferencial Evolutivo
diffEvo <- function(FUN,lb,ub,N)
{
  #- FUN = função objetivo a ser otimizada
  #- lb = vetor contendo os limites superiores das variáveis de decisão
  #- N = tamanho da população inicial
  
  #-----------------Definição dos parâmetros
  #Geração atual
  g = 1   
  #Máximo número de gerações
  gmax = 40
  
  #Probabilidade de recombinação
  c = 0.7
  #Fator de escala
  f = 0.99
  l = 0.5
  
  #Dimensão do problema
  n = length(lb)
  
  #Parâmetro de controle
  bestIndTroughTime = list()
  
  #População inicial
  P = matrix(0,nrow=N,ncol=n)
  for(i in 1:n)
  {
    P[,i]<-runif(n = N,min = lb[i],max = ub[i])
  }
  
  #Matriz de recombinação
  U = matrix(0,nrow=N,ncol=n)
  
  #Avaliação das soluções candidatas
  jP = evalFobj(FUN,P)
  
  #Vetores de controle de população; 
  Jbest = c()
  Jmed = c()
  
  bench = 1
  mov.ind = list()
  
  #------------------LOOP Principal
  while(g<=gmax && (sum(bench)!=0))
  {
    
    #Calcula tempo de iteração
    t0 = Sys.time()
    
    #Calcula melhores indivíduos
    Jbest[g]=min(jP)
    Jmed[g]=mean(jP)
    
    #Recombinação dos indivíduos
    bench = c()
    for(i in 1:N)
    {
      r = c(sample(N,3),which.min(jP))
      delta = sample(N,1)
      for(j in 1:n)
      {
        if(runif(1)<=c || j==delta)
        {
          U[i,j] = P[r[1],j] + f*(P[r[2],j]-P[r[3],j]) + l*((P[r[1],j]-P[r[4],j]))
        }else
        {
          U[i,j] = P[i,j]
        }
      }
      
      bench[i] = ((P[r[1],j]-P[r[4],j]))
      mov.ind[[g]] = bench
      
      #Reflexão para dentro dos limites factíveis
      for(j in 1:n)
      {
        if(U[i,j]<lb[j])
        {
          U[i,j] = lb[j]
        }else if(U[i,j]>ub[j])
        {
          U[i,j] = ub[j]
        }
      }
      
      #Seleção dos indivíduos
      if(FUN(U[i,]) <= jP[i])
      {
        P[i,] = U[i,]
      }
    }
    
    #Avaliação das soluções candidatas
    jP = evalFobj(FUN,P)
    
    #Printa iteração e tempo
    print(Sys.time()-t0)
    print(c(g,gmax,Jbest[g]))
    print(P[which.max(jP),])
    
    #Melhor índividuo da geração
    bestIndTroughTime[[g]] <- P[which.max(jP),]
    
    #Iteração
    g = g + 1
  }
  
  output = list(Jbest = Jbest,
                Jmed = Jmed,
                best.ind = P[which.min(jP),],
                best.fitness = min(jP),
                POP = P,
                dif.vectors = mov.ind,
                jP = jP,
                bestIndTroughTime = bestIndTroughTime)
  
  return(output)
}

#Avaliação da função objetivo p/ toda população
evalFobj <- function(FUN,P)
{
  N = nrow(P)
  n = ncol(P)
  
  output = c()
  for(i in 1:N)
  {
    output[i] = FUN(P[i,])
  }
  
  return(output)
}

#Plota evolução da população
plotFobj <- function(x)
{
  #Labels p/ legenda
  Legenda = c()
  Legenda[1:length(x[[1]])] = "Jbest"
  Legenda[(length(x[[1]])+1):(length(x[[1]])+length(x[[2]]))] = "Jmed"
  
  
  #Dataframe para ggplot
  df = data.frame(
    Generations = (1:length(x[[1]])),
    Fitness = c(x[[1]],x[[2]]),
    Legend = Legenda
  )
  
  #Plot das funções objetivo
  ggplot(data=df, aes(x=Generations, y=Fitness, color=Legend)) +
    geom_line(size=0.8) +
    theme(legend.position="bottom",plot.title = element_text(hjust = 0.5)) + 
    ggtitle("Evolução da população") + 
    geom_hline(aes(yintercept=-1),
             linetype=4, colour="black")
  
}

#Preenchimento de NAs
replaceNAsByMedian <- function(X)
{
  for(i in 1:ncol(X))
  {
    X[is.na(X[,i]),i] <- median(na.omit(X[,i])) 
  }
  return(X)
}

#Define classes
decodeClass <- function(X)
{
  X = as.factor(X)
  n = length(levels(X))
  if(n<=2)
  {
    Y = c()
    Y[which(X==levels(X)[1])] = 1
    Y[which(X==levels(X)[2])] = -1
  }else
  {
    Y = matrix(-1,nrow(nrow(X)),ncol=n)
    for(i in 1:n)
    {
      Y[which(X==levels(X)[i]),i]<--1
    }
  }
  return(Y)
}

#Função Objetivo
#OBS: RETORNA -ACC PARA REALIZAR A OTIMIZAÇÃO
trainAndTestSVM.OBJ <- function(x)
{
  
  #Definição dos parâmetros
  C = 10^x[1]
  g = 10^x[2]
  
  #Treino e teste
  splitIndex = createDataPartition(Y,times=1,p=0.7,list=FALSE)
  trainX = X[splitIndex,]
  testX = X[-splitIndex,]
  trainY = Y[splitIndex]
  testY = Y[-splitIndex]
  
  #Treina e constrói o modelo
  svm.model = svm(trainY ~ ., data=trainX,
                  cost=C,gamma=g)
  
  #Prediz a classe das amostras de teste
  Yhat = as.numeric(predict(svm.model,testX))
  
  #Aplica o limiar
  Yhat[Yhat<0] = -1
  Yhat[Yhat>=0] = 1
  
  #Caucula a AUC e ACC
  #acc<-sum(diag(table(Yhat,testY)))/sum(table(Yhat,testY))
  auc = AUC::auc(AUC::roc(Yhat,factor(testY)))
  
  #Retorna -AUC para a otimização
  out = -auc
  
  return(out)
}

#Treina e testa SVM genérico
trainAndTestSVM <- function(trainX,trainY,testX,testY,C,g)
{
  #Treina e constrói o modelo
  svm.model = svm(trainY ~ ., data=trainX,
                  cost=C,gamma=g)
  
  #Prediz a classe das amostras de teste
  Yhat = as.numeric(predict(svm.model,testX))
  
  #Aplica o limiar
  Yhat[Yhat<0] = -1
  Yhat[Yhat>=0] = 1
  
  #Caucula a AUC e ACC
  #acc<-sum(diag(table(Yhat,testY)))/sum(table(Yhat,testY))
  auc = AUC::auc(AUC::roc(Yhat,factor(testY)))
  
  #Retorna -AUC para a otimização
  out<- auc
  
  return(out)
}

#Validação cruzada
crossValidation <- function(trainX,trainY,Crange,gammarange)
{
  idx = createFolds(1:nrow(trainX), k = 10)
  acc.svm = matrix(0, nrow=length(gammarange),ncol=length(Crange))
  for(i in 1:length(gammarange))
  {
    for(j in 1:length(Crange))
    {
      acc<-c()
      auc<-c()
      t0<-Sys.time()
      for(k in 1:10)
      {
        acc[k] <- trainAndTestSVM(trainX = trainX[-idx[[k]],],
                                  trainY = trainY[-idx[[k]]],
                                  testX = trainX[idx[[k]],],
                                  testY = trainY[idx[[k]]],
                                  C = Crange[j],g = gammarange[i])
      }
      acc.svm[i,j]<-mean(acc)
      print(Sys.time()-t0)
      print(c("Internal LOOP:",j,"External LOOP:",i))
      print(c(acc.svm[i,j]))
    }
    
    return(acc.svm)
    
  }
  
  output <- list(AUC = auc.svm, ACC = acc.svm)
  return(output)
  
}

#------------------------------------Definição dos dados
#data("BreastCancer")
data("PimaIndiansDiabetes")
#data("Ionosphere")
#BaseCar <- read_csv("~/Reconhecimento de Padrões/Desbalanceamento/BaseCar.csv")

#Dados de entrada
#X <- data.matrix(BreastCancer[,2:10])
X <- data.matrix(PimaIndiansDiabetes[,1:8]) 
#X <- data.matrix(Ionosphere[,1:34])
#X <- data.matrix(BaseCar[,2:7])
X <- replaceNAsByMedian(X[,-2])
X <- data.Normalization(X,type="n1")

#Dados de saída
Y <- decodeClass(PimaIndiansDiabetes[,9])
#Y <- decodeClass(Ionosphere[,35])
#Y <- decodeClass((as.factor(data.matrix(BaseCar[,8]))))

#-----------------------------------Otimização dos parâmetros da SVM

#Limites para os parâmetros c(Gamma,Cost)
# gammarange <- c(2 %o% 10^(-12:4))
# Crange <- c(2 %o% 10^(-5:15))
Llim<-c(-5,-5)
Ulim<-c(5,5)

#Execução da otimização
DE.out <- diffEvo(trainAndTestSVM.OBJ,Llim,Ulim,30)

#Plot da evolução da população
plotFobj(DE.out)

#Plot da distribuição final da população
plot(DE.out[[5]],xlim=c(Llim[1],Ulim[1]),ylim=c(Llim[2],Ulim[2]),
     xlab='log(cost)',ylab='log(gamma)',main='Distribuição final da população')
par(new=T)
plot(DE.out[[3]][1],DE.out[[3]][2],xlim=c(Llim[1],Ulim[1]),ylim=c(Llim[2],Ulim[2]),
     xlab=' ',ylab=' ',main=' ',col='red')

#---------------------------------------Avaliação dos resultados
#Histograma para o parâmetro C
qplot(10^DE.out[[5]][,1], geom="histogram",xlab='Cost',
      main='Cost Histogram') 

#Histograma para o parâmetro gamma
qplot(10^DE.out[[5]][,2], geom="histogram",xlab='Gamma',
      main='Gamma Histogram') 

#Posições do melhor indivíduo
plot(matrix(unlist(DE.out$bestIndTroughTime),ncol=2,byrow=T),type='b',
     xlab='log(gamma)',ylab='log(C)',
     main='Posição do melhor indivíduo ao longo das gerações',
     xlim=c(Llim[1],Ulim[1]),ylim=c(Llim[2],Ulim[2]))
par(new=T)
plot(matrix(unlist(DE.out$bestIndTroughTime),ncol=2,byrow=T)[1,1],
     matrix(unlist(DE.out$bestIndTroughTime),ncol=2,byrow=T)[1,2],
     col='red',xlab=' ',ylab=' ',main=' ',
     xlim=c(Llim[1],Ulim[1]),ylim=c(Llim[2],Ulim[2]),pch=4)
par(new=T)
plot(matrix(unlist(DE.out$bestIndTroughTime),ncol=2,byrow=T)[40,1],
     matrix(unlist(DE.out$bestIndTroughTime),ncol=2,byrow=T)[40,2],
     col='blue',xlab=' ',ylab=' ',main=' ',
     xlim=c(Llim[1],Ulim[1]),ylim=c(Llim[2],Ulim[2]),pch=19)


C.best <- 10^(DE.out[[3]][1])
gamma.best <- 10^(DE.out[[3]][2])

#--------------------------------------- Seleção por CV com Caret
ctrl <- trainControl(method = "cv", savePred=T, classProb=T)
mod <- caret::train(diabetes ~., data=PimaIndiansDiabetes, 
             method = "svmLinear2", trControl = ctrl)

caret.cost <- mod$finalModel$cost
caret.gamma <- mod$finalModel$gamma

#--------------------------------------- Comparação entre os métodos

acc.final <- c()
acc.final.caret <- c()
for(i in 1:30)
{
  splitIndex <- createDataPartition(Y,times=1,p=0.7,list=FALSE)
  trainX <- X[splitIndex,]
  testX <- X[-splitIndex,]
  trainY <- Y[splitIndex]
  testY <- Y[-splitIndex]
  
  #Acurácia de classificação DE
  acc.final[i] <- trainAndTestSVM(trainX,trainY,testX,
                                  testY,C.best,gamma.best) 
  
  #Acurácia de classificação CV Caret
  acc.final.caret[i] <- trainAndTestSVM(trainX,trainY,testX,
                                  testY,caret.cost,caret.gamma) 
}

print(mean(acc.final))
print(mean(acc.final.caret))

#Tempo de execução
t<-Sys.time()-t
print(t)

#----TESTE DO ALGORITMO----
# k <- diffEvo(fobj,c(-3,-3),c(3,3),100)
# plotFobj(k)
#
# seqi <- seq(-3,3,0.1)
# seqj <- seq(-3,3,0.1)
#
# M <- matrix(0, nrow=length(seqi), ncol=length(seqj))
# for(i in 1:length(seqi))
# {
#   for(j in 1:length(seqj))
#   {
#     xt <- c(seqi[i],seqj[j])
#     M[i,j] <- fobj(xt)
#   }
# }
#
# persp3d(seqi,seqj,
#         alpha=0.9,M,col='lightblue')
#
# print(min(M))
