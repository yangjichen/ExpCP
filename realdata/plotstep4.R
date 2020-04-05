library(ggplot2)

setwd('/Users/yangjichen/Desktop/project/Pub_opi_tensor/exponential\ family\ tensor/code/realdata')
data =read.csv('Effects of Aux Info(222).csv',header = F)

re = c(as.numeric(data[1,]),as.numeric(data[2,]))
MR = rep(seq(0.1,0.8,0.1),2)
category = c(rep('with Aux',8),rep('without Aux',8))

mydata = data.frame(re,MR,category)

ggplot(mydata, aes(x=MR, y=re, color=category,shape=category)) +
  theme(legend.position = c(0.15,0.85))+
  geom_line() +geom_point(size=2)

ggsave('With&Without Aux.png',width = 5, height = 5,dpi=300)
