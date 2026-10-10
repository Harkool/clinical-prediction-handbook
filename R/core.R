# Core functions for the independent synthetic case A (no contributed packages).
# Source this file, then predict_case(readRDS("results/model.rds"), newdata).

case_recipe <- function() list(
  age_knots = c(20, 45, 65, 85),
  predictors = c("age", "bmi", "male", "asa", "emergency", "diabetes",
                 "copd", "smoker", "chf", "sbp", "hr", "hb", "alb", "crea"),
  parameters = 18L, version = "A-2026-10-10")

rcs_age <- function(x, knots) {
  positive_cube <- function(z) pmax(z, 0)^3
  last <- length(knots); a <- knots[last - 1]; b <- knots[last]
  nonlinear <- do.call(cbind, lapply(knots[seq_len(last - 2)], function(k)
    (positive_cube(x-k) - positive_cube(x-a)*(b-k)/(b-a) +
       positive_cube(x-b)*(a-k)/(b-a))/((b-knots[1])^2*10)))
  cbind(age_linear=(x-50)/10, age_rcs1=nonlinear[,1], age_rcs2=nonlinear[,2])
}

make_design <- function(d, recipe=case_recipe()) {
  missing <- setdiff(recipe$predictors, names(d))
  if (length(missing)) stop("Missing inputs: ", paste(missing, collapse=", "))
  if (anyNA(d[recipe$predictors])) stop("Case A requires complete inputs; use a separately validated missing-input strategy.")
  binary <- c("male","emergency","diabetes","copd","smoker","chf")
  if (any(!as.matrix(d[binary]) %in% c(0,1))) stop("Binary inputs must be 0/1")
  if (any(!d$asa %in% 1:4) || any(d$crea <= 0)) stop("Invalid ASA or creatinine")
  X <- cbind("(Intercept)"=rep(1, nrow(d)), rcs_age(d$age, recipe$age_knots),
    bmi=(d$bmi-25)/5, male=d$male,
    asa_II=as.integer(d$asa==2), asa_III=as.integer(d$asa==3),
    asa_IV=as.integer(d$asa==4), emergency=d$emergency,
    diabetes=d$diabetes, copd=d$copd, smoker=d$smoker, chf=d$chf,
    sbp=(d$sbp-120)/20, hr=(d$hr-75)/15, hb=(d$hb-130)/20,
    alb=(d$alb-40)/5, log_crea=log(d$crea/80))
  storage.mode(X) <- "double"
  if (any(!is.finite(X))) stop("Non-finite design values")
  X
}

checked_binomial_fit <- function(X,y,offset=NULL) {
  if(anyNA(y) || any(!y %in% 0:1) || length(unique(y))<2L)
    stop("The fit needs two observed outcome classes")
  warnings <- character()
  fit <- withCallingHandlers(glm.fit(X,y,family=binomial(),offset=offset,
    control=glm.control(epsilon=1e-10,maxit=100)),warning=function(w) {
      warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")
    })
  gradient <- max(abs(crossprod(X,fit$fitted.values-y)))/nrow(X)
  if(length(warnings) || !isTRUE(fit$converged) || isTRUE(fit$boundary) ||
     fit$rank != ncol(X) || any(!is.finite(fit$coefficients)) || gradient>1e-7 ||
     any(fit$fitted.values<=0 | fit$fitted.values>=1))
    stop("Fit failed convergence/rank/boundary/warning/gradient/probability checks")
  # These numerical checks are not a complete separation diagnostic.
  fit$gradient <- gradient
  fit
}

fit_case <- function(d, recipe=case_recipe()) {
  X <- make_design(d, recipe)
  if (anyNA(d$y) || any(!d$y %in% 0:1) || length(unique(d$y))<2)
    stop("The primary fit needs two observed outcome classes")
  fit <- checked_binomial_fit(X,d$y)
  gradient <- fit$gradient
  list(fit=fit, recipe=recipe,
       target=list(outcome="30-day CD >= III, including death (CD V)",
                   decision_time="pre-anaesthetic assessment complete, before induction",
                   outcome_origin="actual start of surgery", window_days=30),
       metadata=list(version=recipe$version, n=nrow(d), events=sum(d$y),
                     parameters=ncol(X)-1L, gradient=gradient))
}

predict_case <- function(model, newdata) {
  X <- make_design(newdata, model$recipe)
  if (!identical(colnames(X), names(model$fit$coefficients))) stop("Model/recipe mismatch")
  as.vector(plogis(X %*% model$fit$coefficients))
}

auc <- function(y, p) {
  n1 <- sum(y==1); n0 <- sum(y==0)
  if (!n1 || !n0) return(NA_real_)
  (sum(rank(p, ties.method="average")[y==1])-n1*(n1+1)/2)/(n1*n0)
}

performance <- function(y, p) {
  p <- pmin(pmax(p, 1e-12), 1-1e-12)
  lp <- qlogis(p)
  cal <- tryCatch(checked_binomial_fit(cbind(1,lp),y),error=function(e) NULL)
  citl <- tryCatch(checked_binomial_fit(matrix(1,length(y),1),y,offset=lp),error=function(e) NULL)
  slope <- if (!is.null(cal)) unname(cal$coefficients[2]) else NA_real_
  intercept <- if (!is.null(citl)) unname(citl$coefficients[1]) else NA_real_
  c(C=auc(y,p), Brier=mean((p-y)^2), calibration_slope=slope,
    calibration_intercept=intercept, OE=sum(y)/sum(p))
}

net_benefit <- function(y, p, thresholds) sapply(thresholds, function(t) {
  decision <- p >= t
  mean(decision*y)-t/(1-t)*mean(decision*(1-y))
})

bootstrap_optimism <- function(d, model, B=200L, seed=20261011L,
                               thresholds=c(.05,.10,.15,.20,.25,.30)) {
  set.seed(seed)
  apparent <- performance(d$y, predict_case(model,d))
  nb0 <- net_benefit(d$y,predict_case(model,d),thresholds)
  delta <- matrix(NA_real_,B,length(apparent),dimnames=list(NULL,names(apparent)))
  nbdelta <- matrix(NA_real_,B,length(thresholds))
  for (b in seq_len(B)) {
    idx <- sample.int(nrow(d),nrow(d),replace=TRUE)
    boot <- d[idx,,drop=FALSE]
    # Repeat the entire prespecified process: recipe -> design -> ML fit.
    # There is no imputation, selection or tuning in case A.
    mb <- fit_case(boot,model$recipe)
    pb <- predict_case(mb,boot); po <- predict_case(mb,d)
    delta[b,] <- performance(boot$y,pb)-performance(d$y,po)
    if(any(!is.finite(delta[b,]))) stop("Calibration/performance failure in development bootstrap iteration ",b)
    nbdelta[b,] <- net_benefit(boot$y,pb,thresholds)-net_benefit(d$y,po,thresholds)
  }
  optimism <- colMeans(delta)
  table <- data.frame(metric=names(apparent), apparent=unname(apparent),
    mean_optimism=unname(optimism), corrected=unname(apparent-optimism),
    optimism_MCSE=apply(delta,2,sd)/sqrt(B), B=B, row.names=NULL)
  # Corrected O/E is not interpreted as an independent external calibration estimate.
  dca <- data.frame(threshold=thresholds, apparent_NB=nb0,
    optimism=colMeans(nbdelta), corrected_NB=nb0-colMeans(nbdelta),
    treat_all=mean(d$y)-(1-mean(d$y))*thresholds/(1-thresholds), treat_none=0,
    optimism_MCSE=apply(nbdelta,2,sd)/sqrt(B))
  list(table=table, replicates=delta, dca=dca)
}

fixed_model_intervals <- function(y,p,B=500L,seed=20261012L) {
  set.seed(seed)
  draws <- replicate(B, {i <- sample.int(length(y),length(y),replace=TRUE);
                         performance(y[i],p[i])})
  estimate <- performance(y,p)
  interval <- t(apply(draws,1,quantile,probs=c(.025,.975),na.rm=TRUE))
  data.frame(metric=names(estimate), estimate=unname(estimate),
    lower=interval[,1],upper=interval[,2],valid_replicates=rowSums(is.finite(draws)),
    method="conditional empirical bootstrap; fixed model; apparent evaluation",
    row.names=NULL)
}

subgroup_performance <- function(d,p,B=300L,seed=20261013L) {
  set.seed(seed)
  groups <- c(list(all=rep(TRUE,nrow(d))),
    setNames(lapply(0:1,function(k)d$male==k),paste0("male_",0:1)),
    setNames(lapply(1:6,function(k)d$center==k),paste0("center_",1:6)))
  rows <- lapply(names(groups),function(g) {
    i <- which(groups[[g]]); y <- d$y[i]; pg <- p[i]
    estimate <- auc(y,pg)
    draw <- if(length(i)) replicate(B, {j <- sample.int(length(i),length(i),TRUE);
      auc(y[j],pg[j])}) else rep(NA_real_,B)
    valid <- sum(is.finite(draw))
    ci <- if(valid>=100) quantile(draw,c(.025,.975),na.rm=TRUE) else c(NA_real_,NA_real_)
    data.frame(group=g,n=length(i),events=sum(y),non_events=sum(y==0),C=estimate,
      lower=ci[1],upper=ci[2],valid_replicates=valid,
      conclusion=if(valid<100)"interval not estimable reliably in this resampling run"
        else if(sum(y)<30)"imprecise descriptive result; no subgroup superiority claim"
        else "descriptive within-development-sample result",row.names=NULL)
  })
  do.call(rbind,rows)
}

make_state <- function(d, data_cut, window_days=30,
                       outcome=c("composite_CD","nonfatal_complication")) {
  outcome <- match.arg(outcome)
  need <- c("outcome_origin","event_time","death_time","last_contact")
  if (!all(need %in% names(d))) stop("Missing outcome timestamps")
  # Here times are numeric days on a common scale; production records need
  # adjudication of timestamp precision, occurrence time and ascertainment time.
  origin <- d$outcome_origin; end <- origin+window_days
  if(anyNA(origin) || anyNA(d$last_contact) || anyNA(data_cut))
    stop("Observation limits must be known; missing follow-up cannot prove no event")
  ev <- d$event_time; death <- d$death_time
  impossible <- (!is.na(ev)&ev<origin) | (!is.na(death)&death<origin) |
    (!is.na(ev)&!is.na(death)&ev>death) |
    (!is.na(d$last_contact)&d$last_contact<origin)
  if(any(impossible)) stop("Impossible event/death/follow-up chronology at rows: ",
                          paste(which(impossible),collapse=","))
  observed_end <- pmin(d$last_contact,data_cut,end,na.rm=TRUE)
  known_event <- !is.na(ev)&ev<=observed_end
  known_death <- !is.na(death)&death<=observed_end
  if(outcome=="composite_CD") known_event <- known_event|known_death
  state <- ifelse(known_event,"event",
    ifelse(outcome=="nonfatal_complication" & known_death,"competing",
      ifelse(observed_end>=end,"no_event_complete",
        ifelse(data_cut<end & data_cut<=d$last_contact,"incomplete_admin",
               "incomplete_lost"))))
  y <- ifelse(state=="event",1L,
    ifelse(state %in% c("no_event_complete","competing"),0L,NA_integer_))
  data.frame(state=state,y=y,observed_end=observed_end)
}

simulate_case <- function(n=3200L,seed=20261010L,event_rate=.12) {
  set.seed(seed)
  d <- data.frame(id=sprintf("A%05d",seq_len(n)),
    center=sample(1:6,n,TRUE,prob=c(.25,.23,.20,.17,.12,.03)),
    age=pmin(pmax(round(rnorm(n,58,15)),18),90),
    bmi=pmin(pmax(rnorm(n,26,4.5),16),45), male=rbinom(n,1,.55),
    asa=sample(1:4,n,TRUE,prob=c(.16,.47,.30,.07)),
    emergency=rbinom(n,1,.16), diabetes=rbinom(n,1,.21),
    copd=rbinom(n,1,.11),smoker=rbinom(n,1,.25),chf=rbinom(n,1,.07),
    sbp=pmin(pmax(rnorm(n,125,18),75),200),hr=pmin(pmax(rnorm(n,77,13),40),140),
    hb=pmin(pmax(rnorm(n,129,18),70),180),alb=pmin(pmax(rnorm(n,39,5),20),52),
    crea=pmin(pmax(exp(rnorm(n,log(80),.4)),30),400))
  X <- make_design(d)
  beta <- c(0,.30,.30,-.50,.15,.12,.30,.70,1.15,.65,.35,.35,.12,.60,
            -.15,.20,-.25,-.45,.40)
  names(beta) <- colnames(X)
  linear <- as.vector(X[,-1,drop=FALSE]%*%beta[-1])
  beta[1] <- uniroot(function(b)mean(plogis(b+linear))-event_rate,c(-15,5))$root
  true_risk <- plogis(as.vector(X%*%beta))
  d$y <- rbinom(n,1,true_risk)
  d$cd_grade <- ifelse(d$y==1, sample(c(3L,4L,5L),n,TRUE,c(.65,.23,.12)),
                      sample(0:2,n,TRUE,c(.45,.35,.20)))
  d$decision_time <- 20000+runif(n,0,365)
  d$outcome_origin <- d$decision_time+runif(n,2/24,24/24)
  d$collection_time <- d$decision_time-runif(n,2/24,72/24)
  d$result_available_time <- d$decision_time-runif(n,.1/24,1/24)
  d$event_time <- ifelse(d$y==1 & d$cd_grade!=5,
                        d$outcome_origin+runif(n,.5,29.5),NA_real_)
  d$death_time <- ifelse(d$cd_grade==5,d$outcome_origin+runif(n,.5,29.5),NA_real_)
  d$last_contact <- d$outcome_origin+30
  d$true_risk <- true_risk
  list(data=d,true_beta=beta,recipe=case_recipe())
}

write_table <- function(x,path) write.csv(x,path,row.names=FALSE,fileEncoding="UTF-8",na="NA")
