# Title     : 2-stage NLP Statistical Analysis
# Objective : do significance testing for CogSci 2019 paper
# Created by: ph
# Created on: 12/11/18

# use CTL + Numpad2 to pushc code to R console (open R Console from Tools

dat = read.csv(file='/media/lab/2StageNLP/matching.csv', header=TRUE, sep=",", row.names=1)
levels(dat$group)
boxplot(score~stage,data=dat)

# one-way
res.aov1 <- aov(score ~ stage, data = dat)
summary(res.aov1)
TukeyHSD(res.aov1, which="stage")
plot(res.aov1, 1)

# two-way
res.aov2 <- aov(score ~ stage * task, data = dat)
summary(res.aov2)

# use only single task
print('New task')
task_dat = dat[which(dat$task == 'hypernyms'), names(dat) %in% c('task','stage', 'score', 'embedder', 'embed_size')]
res.aov2 <- aov(score ~ stage, data = task_dat)
summary(res.aov2)

# hypernyms does not show significant diff betw stages

# TODO remove expert+rs stage

# TODO what about embedder interaction?