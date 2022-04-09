library(survival)

# set to print 16 digits
options(digits=16)

## Example 1

# set seed
set.seed(0)
n.c <- 20
h.c <- 3.321
n.t <- 30
h.t <- 0.592

# generate data
control <- rexp(n.c, h.c)
treatment <- rexp(n.t, h.t)
df.c <- data.frame(time=control, status=0, type=0)
df.t <- data.frame(time=treatment, status=0, type=1)
df <- rbind(df.c, df.t)
df <- df[order(df$time),]

# suffices to store censor times as each outcome times
censor_times <- df$time
# get output chisq stats
chisqs <- rep(NA, length(censor_times))
for (i in 1:length(censor_times)) {
    df$status[i] = 1
    out = survdiff(Surv(df$time, event=df$status) ~ df$type)
    chisqs[i] = out$chisq
}

## Example 2

# set some values to be the same
df$time[2:4] <- df$time[2]
df$time[9:14] <- df$time[9]
df$time[n.c + (2:6)] <- df$time[n.c + 2]
df$status <- 0

# repeat analysis
censor_times <- sort(unique(df$time))
censor_times_counts <- summary(as.factor(df$time))
chisqs <- rep(NA, length(censor_times))
pos <- 1
for (i in 1:length(censor_times)) {
    df$status[pos + (0:(censor_times_counts[i]-1))] = 1
    print(df[pos + (0:(censor_times_counts[i]-1)),])
    out = survdiff(Surv(df$time, event=df$status) ~ df$type)
    chisqs[i] = out$chisq
    pos = pos + censor_times_counts[i]
}
