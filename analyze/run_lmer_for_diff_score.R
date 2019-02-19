

library(lmerTest)

dat = read.csv(file='/media/lab/2StageNLP/diff_scores.csv', header=TRUE, sep=",", row.names=1)
dat$task = factor(dat$task)

fit <- lmer(diff_score ~ all_probe_sim + (1|job_name), data=dat)
anova(fit)


# scatter
par(cex = 3.0)  # magnify text on plot
plot(x=dat$all_probe_sim, y=dat$diff_score, main="Scatter",
   xlab="Avg Similarity of all pairs in task", ylab="BalAcc Difference (Classifier - Comparator)", pch=19)
fit_lm = lm(diff_score ~ all_probe_sim, data=dat)
summary(fit_lm)
abline(fit_lm, col="red", lwd=5) # regression line (y~x)
abline(h=0, col="grey", lwd=2)
