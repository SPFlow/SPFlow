# Title     : TODO
# Objective : TODO
# Created by: molina
# Created on: 3/21/18

library("histogram")


getHistogram <- function(data, seed=1) {
	set.seed(seed)
	hh <- histogram(data, verbose = FALSE, plot = FALSE, penalty="penR")
	return(hh)
	breaks <- hh$breaks
	return(list("breaks" = breaks))
}