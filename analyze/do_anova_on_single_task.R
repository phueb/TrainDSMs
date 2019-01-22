# Title     : 2-stage NLP Statistical Analysis
# Objective : do significance testing for CogSci 2019 paper
# Created by: ph
# Created on: 12/11/18

# use CTL + Numpad2 to pushc code to R console (open R Console from Tools

dat = read.csv(file='/media/lab/2StageNLP/2stage_data.csv', header=TRUE, sep=",", row.names=1)

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

# use only single task
TASK = 'nyms_ant_jw'
task_dat = dat[which(dat$task == TASK),
    names(dat) %in% c('task','stage', 'score', 'embedder', 'embed_size', 'job_name')]

# plot
par(cex = 1.5)  # magnify text on plot
with(dat, interaction.plot(stage, embedder, score,
  ylim = c(0.5, 1), lwd = 5,
  ylab = "mean of score for" + TASK, xlab = "stage", trace.label = "embedder"))

# anova
task_dat.aov <- aov(score ~ embedder * stage + Error(job_name), data = task_dat)
summary(task_dat.aov)
