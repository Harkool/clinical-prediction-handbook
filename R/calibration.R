# Prespecified smooth calibration and risk distribution for case A.
# Three logit knots (5%,50%,95%): one interior knot and two boundaries.
calibration_fit <- function(y,p,grid,knots) {
  if(length(y)!=length(p) || anyNA(y) || any(!y %in% 0:1) || length(unique(y))<2L ||
     any(!is.finite(p)) || any(p<=0 | p>=1)) stop("Invalid calibration inputs/classes")
  design <- function(prob) cbind(1,splines::ns(qlogis(prob),knots=knots[2],
                                      Boundary.knots=knots[c(1,3)]))
  X <- design(p); warnings <- character()
  fit <- withCallingHandlers(glm.fit(X,y,family=binomial(),
    control=glm.control(epsilon=1e-10,maxit=100)),warning=function(w) {
      warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")
    })
  gradient <- max(abs(crossprod(X,fit$fitted.values-y)))/length(y)
  if(length(warnings) || !isTRUE(fit$converged) || isTRUE(fit$boundary) ||
     fit$rank!=ncol(X) || any(!is.finite(fit$coefficients)) || gradient>1e-7)
    stop("Calibration fit failed convergence/rank/boundary/warning/gradient checks")
  as.vector(plogis(design(grid)%*%fit$coefficients))
}

calibration_groups <- function(y,p,breaks=c(0,.05,.10,.15,.20,.30,.50,1)) {
  groups <- cut(p,breaks,include.lowest=TRUE,right=FALSE)
  stopifnot(!anyNA(groups))
  rows <- lapply(seq_len(length(breaks)-1L),function(k) {
    i <- which(as.integer(groups)==k); n <- length(i); events <- sum(y[i])
    rate <- if(n) events/n else NA_real_; z <- qnorm(.975)
    ci <- if(n) {
      mid <- (rate+z^2/(2*n))/(1+z^2/n)
      half <- z*sqrt(rate*(1-rate)/n+z^2/(4*n^2))/(1+z^2/n)
      c(mid-half,mid+half)
    } else c(NA_real_,NA_real_)
    data.frame(lower=breaks[k],upper=breaks[k+1],n=n,events=events,
      mean_prediction=if(n) mean(p[i]) else NA_real_,event_rate=rate,
      wilson_lower=ci[1],wilson_upper=ci[2],
      scope="within-group apparent event rate; descriptive Wilson interval")
  })
  do.call(rbind,rows)
}

calibration_artifacts <- function(d,model,B=200L,conditional_B=500L,
                                  seed=20261011L,conditional_seed=20261012L) {
  p <- predict_case(model,d)
  knots <- unname(quantile(qlogis(p),c(.05,.50,.95),type=7))
  support <- unname(quantile(p,c(.01,.99),type=7))
  grid <- seq(support[1],support[2],length.out=80L)
  apparent <- calibration_fit(d$y,p,grid,knots)
  delta <- matrix(NA_real_,B,length(grid)); conditional <- matrix(NA_real_,conditional_B,length(grid))
  log <- data.frame(stage=character(),iteration=integer(),status=character(),reason=character())
  record <- function(stage,b,fun) {
    failure <- ""
    value <- tryCatch(fun(),error=function(e) {failure <<- conditionMessage(e); rep(NA_real_,length(grid))})
    log <<- rbind(log,data.frame(stage=stage,iteration=b,
      status=if(nzchar(failure)) "FAILED" else "PASS",reason=failure))
    value
  }
  set.seed(seed)
  for(b in seq_len(B)) {
    idx <- sample.int(nrow(d),nrow(d),replace=TRUE)
    delta[b,] <- record("development",b,function() {
      db <- d[idx,,drop=FALSE]; mb <- fit_case(db,model$recipe)
      calibration_fit(db$y,predict_case(mb,db),grid,knots)-
        calibration_fit(d$y,predict_case(mb,d),grid,knots)
    })
  }
  set.seed(conditional_seed)
  for(b in seq_len(conditional_B)) {
    idx <- sample.int(nrow(d),nrow(d),replace=TRUE)
    conditional[b,] <- record("fixed_apparent",b,function()
      calibration_fit(d$y[idx],p[idx],grid,knots))
  }
  nb <- sum(log$stage=="development" & log$status=="PASS")
  nc <- sum(log$stage=="fixed_apparent" & log$status=="PASS")
  # Conservative policy: no failed-fit deletion followed by a success-only average.
  corrected <- if(nb==B) apparent-colMeans(delta) else rep(NA_real_,length(grid))
  bounds <- if(nc==conditional_B) t(apply(conditional,2,quantile,c(.025,.975)))
    else matrix(NA_real_,length(grid),2)
  curve <- data.frame(prediction=grid,apparent=apparent,corrected=corrected,
    apparent_lower=bounds[,1],apparent_upper=bounds[,2],bootstrap_success=nb,
    conditional_success=nc,optimism_mcse=if(nb==B) apply(delta,2,sd)/sqrt(B) else NA_real_,
    interval_target="fixed_model_apparent_curve_pointwise",interval_method="empirical percentiles; fixed prediction pairs")
  recipe <- data.frame(item=c("logit_knot_05","logit_knot_50","logit_knot_95",
    "support_probability_01","support_probability_99","grid_n","development_B","conditional_B",
    "development_seed","conditional_seed","corrected_curve_interval","failed_fit_policy"),
    value=c(knots,support,length(grid),B,conditional_B,seed,conditional_seed,
      "none; corrected point estimate plus optimism MCSE","any failure invalidates that entire result"))
  list(curve=curve,groups=calibration_groups(d$y,p),recipe=recipe,log=log,
       delta=delta,success=(nb==B && nc==conditional_B),predictions=p)
}

plot_calibration <- function(cal,file,format=c("pdf","png")) {
  format <- match.arg(format); if(!isTRUE(cal$success)) stop("Failed calibration run cannot be plotted as successful")
  if(format=="pdf") pdf(file,width=7.2,height=6.8) else png(file,width=1440,height=1360,res=200)
  on.exit(dev.off(),add=TRUE)
  layout(matrix(1:2,2,1),heights=c(3,1.25)); par(mar=c(3,4,2,1),las=1)
  x <- cal$curve
  plot(x$prediction,x$apparent,type="n",xlim=c(0,1),
    ylim=range(c(0,1,x$corrected)),xlab="",ylab="Observed risk (smooth estimate)")
  polygon(c(x$prediction,rev(x$prediction)),c(x$apparent_lower,rev(x$apparent_upper)),
    col=adjustcolor("#3679B0",alpha.f=.18),border=NA)
  abline(0,1,col="grey60",lty=3)
  lines(x$prediction,x$apparent,col="#3679B0",lwd=2)
  lines(x$prediction,x$corrected,col="#B6531A",lty=2,lwd=2)
  legend("topleft",c("Apparent + pointwise conditional range","Bootstrap corrected (no interval)"),
    col=c("#3679B0","#B6531A"),lty=c(1,2),lwd=2,bty="n",cex=.80)
  title("Case A: calibration on prespecified support")
  par(mar=c(4,4,1,1))
  hist(cal$predictions,breaks=seq(0,1,by=.025),col="#B7C9D8",border="white",
    xlim=c(0,1),main="",xlab="Predicted probability",ylab="Patients",freq=TRUE)
}
