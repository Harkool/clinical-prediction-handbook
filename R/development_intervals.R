# Percentile double bootstrap of the internally corrected development estimate.
# Outer resamples reproduce development sampling. Inner resamples estimate
# optimism within each outer sample. Finite inner/outer budgets imply MC error.
development_intervals <- function(d,B_outer=200L,B_inner=50L,seed=20261014L) {
  set.seed(seed)
  indices <- lapply(seq_len(B_outer),function(i)sample.int(nrow(d),nrow(d),TRUE))
  draws <- matrix(NA_real_,B_outer,5,dimnames=list(NULL,
    c("C","Brier","calibration_slope","calibration_intercept","OE")))
  inner_mcse <- draws
  for (b in seq_len(B_outer)) {
    sample <- d[indices[[b]],,drop=FALSE]
    fit <- fit_case(sample)
    correction <- bootstrap_optimism(sample,fit,B=B_inner,seed=seed+b,
                                    thresholds=c(.05,.10,.15,.20,.25,.30))$table
    draws[b,] <- correction$corrected
    inner_mcse[b,] <- correction$optimism_MCSE
    if(b %% 10==0) {cat("Double bootstrap outer replicate",b,"/",B_outer,"\n"); flush.console()}
  }
  interval <- t(apply(draws,2,quantile,probs=c(.025,.975)))
  list(table=data.frame(metric=colnames(draws),lower=interval[,1],upper=interval[,2],
       B_outer=B_outer,B_inner=B_inner,
       mean_inner_optimism_MCSE=colMeans(inner_mcse),
       method="approximate percentile double bootstrap of internal corrected estimate",
       row.names=NULL),replicates=as.data.frame(draws))
}

development_cache_key <- function(d, B_outer, B_inner, seed) {
  # Hash the actual data and function bodies, independent of the current folder.
  tmp <- tempfile(fileext=".rds")
  on.exit(unlink(tmp), add=TRUE)
  saveRDS(list(data=d, algorithm=lapply(list(development_intervals,
    bootstrap_optimism, fit_case, make_design, rcs_age, checked_binomial_fit,
    performance, auc), body)), tmp, version=3)
  list(hash=unname(tools::md5sum(tmp)), R_version=R.version.string,
       B_outer=as.integer(B_outer), B_inner=as.integer(B_inner), seed=as.integer(seed))
}

cached_development_intervals <- function(d, path, B_outer=200L,
                                        B_inner=50L, seed=20261014L) {
  key <- development_cache_key(d, B_outer, B_inner, seed)
  cached <- if(file.exists(path)) readRDS(path) else NULL
  if(!is.null(cached) && identical(cached$key,key)) {
    cat("Reusing double-bootstrap cache: same data, algorithm, budgets and R version.\n")
    return(cached$result)
  }
  result <- development_intervals(d, B_outer=B_outer, B_inner=B_inner, seed=seed)
  dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
  saveRDS(list(key=key,result=result),path,version=3)
  result
}
