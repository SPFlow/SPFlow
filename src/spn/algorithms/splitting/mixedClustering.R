library(dplyr) # for data cleaning
library(cluster) # for gower similarity and pam

mixedclustering <- function(data, featureTypes, n_clusters, random_state) {
  set.seed(random_state) # for reproducibility
  data = as.data.frame(data)
  
  # set data types by family
  for (i in 1:length(featureTypes)){
    if (featureTypes[i] == "categorical") {
      data[i] <- lapply(data[i], function (x) as.factor(x))
    }else if (featureTypes[i] == "discrete") {
		data[i] <- lapply(data[i], function (x) as.ordered(x))
	}else if (featureTypes[i] == "continuous") {
		data[i] <- lapply(data[i], function (x) as.numeric(x))
	}else{
		stop("Feature type not one of: (categorical, discrete, continuous)")
	}
  }
  
  #glimpse(data)
  gower_dist <- daisy(data, metric = "gower")
  
  pam_fit <- pam(gower_dist, diss = TRUE, k = n_clusters)
  
  pam_results <- pam_fit$clustering
  return(pam_results-1)
}