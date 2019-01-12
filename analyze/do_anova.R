# Title     : 2-stage NLP Statistical Analysis
# Objective : do significance testing for CogSci 2019 paper
# Created by: ph
# Created on: 12/11/18

# use CTL + Numpad2 to pushc code to R console (open R Console from Tools

dat = read.csv(file='matching.csv', header=TRUE, sep=",", row.names=1)
levels(dat$group)

boxplot(score~embedder,data=dat)

res.aov <- aov(score ~ embedder, data = dat)
summary(res.aov)
TukeyHSD(res.aov)

plot(res.aov, 1)