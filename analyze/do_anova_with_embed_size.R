# Title     : 2-stage NLP Statistical Analysis
# Objective : do significance testing for CogSci 2019 paper
# Created by: ph
# Created on: 12/11/18

# use CTL + Numpad2 to pushc code to R console (open R Console from Tools

dat = read.csv(file='/media/lab/2StageNLP/2stage_data.csv', header=TRUE, sep=",", row.names=1)
boxplot(score~stage * task,data=dat)

# repeated measures ANOVA
# Convert variables to factor
dat <- within(dat, {
  embedder <- factor(embedder)  # group
  stage <- factor(stage)  # time
  job_name <- factor(job_name)  # subject id
  embed_size <- factor(embed_size)  # PH
})
# drop levels
dat <- subset(dat, stage != "control")
dat <- subset(dat, task != "cohyponyms_syntactic")
dat <- subset(dat, embedder != "random_normal")
levels(dat$stage)
levels(dat$task)
levels(dat$embedder)


par(cex = 1.5)  # magnify text on plot
with(dat, interaction.plot(stage, embed_size, score,
  ylim = c(0.5, 1), lwd = 5,
  ylab = "mean of score (nyms_syn_jw)", xlab = "stage", trace.label = "embed_size"))

dat.aov <- aov(score ~ embed_size * stage + Error(job_name), data = dat)
summary(dat.aov)
