

dat = read.csv(file='/media/lab/2ProcessNLP/2process_data.csv', header=TRUE, sep=",")

# drop levels
dat = dat[dat$task != "cohyponyms_syntactic",]
dat = dat[dat$embedder != "random_normal",]
dat = dat[dat$arch != "classifier",]
# dat = dat[dat$neg_pos_ratio != 0.0,]  # TODO this excludes novice
summary(dat)

# F test to compare variances of two sampels from normal populations
nov_scores = unlist(dat[dat$regime == "novice",]["score"], use.names=FALSE)
exp_scores = unlist(dat[dat$regime == "expert",]["score"], use.names=FALSE)
var.test(nov_scores, exp_scores, alternative="less")

