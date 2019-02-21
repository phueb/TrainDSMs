

library(lmerTest)
# library(psycho)

dat = read.csv(file='/media/lab/2StageNLP/2stage_data.csv', header=TRUE, sep=",")

# drop levels
dat = dat[dat$task != "cohyponyms_syntactic",]
dat = dat[dat$embedder != "random_normal",]
# dat = dat[dat$arch != "classifier",]
# dat = dat[dat$neg_pos_ratio != 0.0,]  # TODO this excludes novice
dat = dat[dat$stage != "control",]
summary(dat)


# mixed-effects model
fit <- lmer(score ~ stage * embedder + (stage|job_name), data=dat)
anova = anova(fit)
summary(fit)
# analyze(anova)
