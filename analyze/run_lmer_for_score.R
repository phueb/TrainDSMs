

library(lmerTest)
library(psycho)
library(forcats)

dat = read.csv(file='/media/research_data/2ProcessNLP/2process_data.csv', header=TRUE, sep=",")


# to factor
dat$num_epochs_per_row_word = factor(dat$num_epochs_per_row_word)
dat$neg_pos_ratio = factor(dat$neg_pos_ratio)
dat$regime = factor(dat$regime)
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
dat = dat[dat$regime != "control",]
print(summary(dat))

# mixed-effects model
fit <- lmer(score ~ regime * arch + (regime|task) + (regime|embedder/job_name) , data=dat)
anova = anova(fit)
print(summary(fit))
print(analyze(anova))
