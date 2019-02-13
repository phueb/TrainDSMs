# Title     : 2-stage NLP Statistical Analysis
# Objective : do significance testing for CogSci 2019 paper
# Created by: ph
# Created on: 12/11/18

# use CTL + Numpad2 to pushc code to R console (open R Console from Tools

dat = read.csv(file='/media/lab/2StageNLP/diff_scores.csv', header=TRUE, sep=",", row.names=1)

# repeated measures ANOVA
# Convert variables to factor
dat <- within(dat, {
  job_name <- factor(job_name)  # subject id

})
# levels
dat <- subset(dat, embedder != "random_normal")
# levels(dat$task)
# levels(dat$embedder)



par(cex = 1.5)  # magnify text on plot
with(dat, interaction.plot(embedder, task, diff_score,
  ylim = c(-0.2, 0.2), lwd = 5,
  ylab = "mean of diff_score", xlab = "embedder", trace.label = "task"))

# ANOVA
aov <- aov(diff_score ~ embedder + task + Error(job_name), data = dat)
TukeyHSD(aov, "task")
summary(aov)


