

library(lmerTest)
library(psycho)
library(forcats)

dat = read.csv(file='/media/lab/2StageNLP/2stage_data.csv', header=TRUE, sep=",")


# to factor
dat$num_epochs_per_row_word = factor(dat$num_epochs_per_row_word)
dat$neg_pos_ratio = factor(dat$neg_pos_ratio)
dat$stage = factor(dat$stage)
dat$arch = factor(dat$arch)
dat$task = factor(dat$task)
dat$evaluation = factor(dat$evaluation)
dat$embed_size = factor(dat$embed_size)
dat$embedder = factor(dat$embedder)
dat$num_vocab = factor(dat$num_vocab)
dat$corpus = factor(dat$corpus)
dat$job_name = factor(dat$job_name)

# drop levels
dat = dat[dat$task != "cohyponyms_syntactic",]
dat = dat[dat$embedder != "random_normal",]
# dat = dat[dat$arch != "classifier",]
dat = dat[(dat$neg_pos_ratio == 1) | (is.na(dat$neg_pos_ratio)), ]
dat = dat[(dat$num_epochs_per_row_word != 0) | (is.na(dat$num_epochs_per_row_word)), ]
dat = dat[dat$stage != "control",]
print(summary(dat))

# mixed-effects model
fit <- lmer(score ~ stage * embedder + (stage|job_name) + (stage|task), data=dat)
anova = anova(fit)
print(summary(fit))
print(analyze(anova))
