# Created by: alejomc
# Created on: 29.04.18
library("Rlab")


set.seed(123)
n <- 1000000
rs <- rnorm(n, mean = 10, sd = 3)
write.csv(rs, row.names = FALSE, col.names = NA, "/Users/alejomc/PycharmProjects/SPNs_Heterogeneous/spn/tests/parametric_samples/norm_mean10_sd3.csv")

set.seed(123)
rs <- rgamma(n, shape = 2, scale = 1/2)
write.csv(rs, row.names = FALSE, col.names = NA, "/Users/alejomc/PycharmProjects/SPNs_Heterogeneous/spn/tests/parametric_samples/gamma_shape2_scale0.5.csv")

set.seed(123)
rs <- rlnorm(n, meanlog = 10, sdlog = 3)
write.csv(rs, row.names = FALSE, col.names = NA, "/Users/alejomc/PycharmProjects/SPNs_Heterogeneous/spn/tests/parametric_samples/lnorm_meanlog_10_sdlog_3.csv")

set.seed(123)
rs <- rexp(n, rate = 2)
write.csv(rs, row.names = FALSE, col.names = NA, "/Users/alejomc/PycharmProjects/SPNs_Heterogeneous/spn/tests/parametric_samples/exp_rate_2.csv")

set.seed(123)
rs <- rpois(n, lambda = 3)
write.csv(rs, row.names = FALSE, col.names = NA, "/Users/alejomc/PycharmProjects/SPNs_Heterogeneous/spn/tests/parametric_samples/pois_lambda_3.csv")

set.seed(123)
rs <- rbern(n, prob = 0.7)
write.csv(rs, row.names = FALSE, col.names = NA, "/Users/alejomc/PycharmProjects/SPNs_Heterogeneous/spn/tests/parametric_samples/bern_prob0.7.csv")

set.seed(123)
rs <- rgeom(n, prob = 0.7)
write.csv(rs, row.names = FALSE, col.names = NA, "/Users/alejomc/PycharmProjects/SPNs_Heterogeneous/spn/tests/parametric_samples/geom_prob0.7.csv")
