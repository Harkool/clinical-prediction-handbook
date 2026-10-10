# Executable R lessons for chapters 01--20. All generated files stay in output_dir.
# Source core.R, interface.R, calibration.R, verify.R, development_intervals.R,
# and competing_branch.R before this file, or use lesson_cli.R.
course_source_repo <- local({
  source_files <- unlist(lapply(sys.frames(),function(frame)
    if(is.character(frame$ofile)) frame$ofile else NULL),use.names=FALSE)
  if(length(source_files)) dirname(dirname(normalizePath(tail(source_files,1),
    winslash="/",mustWork=TRUE))) else normalizePath(getwd(),winslash="/",mustWork=TRUE)
})

course_budgets <- function(mode=c("quick","book")) {
  mode <- match.arg(mode)
  if(mode=="book") list(development=200L,conditional=500L,curve=200L,
    curve_conditional=500L,outer=200L,inner=50L,subgroup=300L)
  else list(development=20L,conditional=40L,curve=20L,
    curve_conditional=40L,outer=10L,inner=10L,subgroup=40L)
}

course_titles <- c("使用说明与总原则","项目、环境与运行记录","合成教学数据",
  "六要素规格与队列","结局、竞争事件与未知标签","可得性与防泄漏",
  "参数预算与缺失边界","回归拟合与显式单位","独立竞争风险 B 分支",
  "实际外层模型比较","训练内调参与嵌套验证","内部验证",
  "校准、风险分布与决策曲线","独立合成验证与更新","亚组与敏感性",
  "会失败的反例检查","预测接口与完整身份","报告材料",
  "投稿准备练习","AI 留痕与人工审核")

course_write_text <- function(x,path) {
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  writeLines(enc2utf8(x),path,useBytes=TRUE)
  invisible(path)
}

course_write_csv <- function(x,path) {
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  write_table(x,path); invisible(path)
}

course_negative <- function(name,fun) {
  reason <- ""
  failed <- tryCatch({fun(); FALSE},error=function(e) {
    reason <<- conditionMessage(e); TRUE
  })
  data.frame(check=name,status=if(failed) "PASS_EXPECTED_REJECTION" else "FAIL_ACCEPTED_INVALID_INPUT",
             observed_error=reason,stringsAsFactors=FALSE)
}

course_data_hash <- function(d,directory) {
  path <- tempfile("course-data-",tmpdir=directory,fileext=".rds")
  on.exit(unlink(path),add=TRUE)
  saveRDS(d,path,version=3,compress=FALSE)
  unname(tools::md5sum(path))
}

course_performance <- function(y,p) {
  if((!is.numeric(y) && !is.logical(y)) || length(y)!=length(p) || length(y)<2L ||
     anyNA(y) || any(!y %in% 0:1) || length(unique(y))!=2L)
    stop("Course evaluation requires two observed binary classes")
  if(!is.numeric(p) || any(!is.finite(p)) || any(p<0 | p>1))
    stop("Course evaluation requires finite probabilities in [0,1]")
  value <- performance(y,p)
  if(any(!is.finite(value)))
    stop("Course performance/calibration metric estimation failed; chapter cannot pass")
  value
}

course_validate_mode <- function(output_dir,mode) {
  path <- file.path(output_dir,"mode.csv")
  if(file.exists(path)) {
    recorded <- read.csv(path,stringsAsFactors=FALSE)$mode
    if(length(recorded)!=1L || is.na(recorded) || recorded!=mode)
      stop("Output directory belongs to a different or invalid mode; choose a separate directory")
  }
  invisible(TRUE)
}

course_chapter_number <- function(chapter) {
  if(length(chapter)!=1L || is.na(chapter) ||
     !grepl("^(0?[1-9]|1[0-9]|20)$",as.character(chapter)))
    stop("R course chapter must be an integer from 1 to 20")
  as.integer(chapter)
}

course_write_status <- function(chapter,output_dir,mode,state,reason="") {
  here <- file.path(output_dir,"chapters",sprintf("ch%02d",chapter))
  dir.create(here,recursive=TRUE,showWarnings=FALSE)
  artifacts <- list.files(here,recursive=TRUE,full.names=TRUE)
  row <- data.frame(chapter=chapter,title=course_titles[chapter],mode=mode,status=state,
    output=substring(here,nchar(output_dir)+2L),artifacts=length(artifacts),
    scope=if(state=="FAIL") paste0("FAILED: ",reason) else
      "synthetic executable exercise; see lesson-specific limitations",stringsAsFactors=FALSE)
  failure_path <- file.path(here,"failure.txt")
  if(state=="FAIL") course_write_text(reason,failure_path)
  else if(file.exists(failure_path)) unlink(failure_path)
  course_write_csv(row,file.path(here,"status.csv"))
  path <- file.path(output_dir,"manifest.csv")
  previous <- if(file.exists(path)) read.csv(path,stringsAsFactors=FALSE) else row[FALSE,]
  previous <- previous[previous$chapter!=chapter,names(row),drop=FALSE]
  manifest <- rbind(previous,row); manifest <- manifest[order(manifest$chapter),]
  course_write_csv(manifest,path)
  invisible(row)
}

course_context <- function(output_dir,mode) {
  mode <- match.arg(mode,c("quick","book"))
  dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
  output_dir <- normalizePath(output_dir,winslash="/",mustWork=TRUE)
  for(dir in c("data","results","cache","chapters","protocol"))
    dir.create(file.path(output_dir,dir),showWarnings=FALSE)
  budget <- course_budgets(mode)
  mode_path <- file.path(output_dir,"mode.csv")
  course_validate_mode(output_dir,mode)
  course_write_csv(data.frame(mode=mode,synthetic=TRUE,n=3200L,seed=20261010L,
    development_B=budget$development,conditional_B=budget$conditional,
    curve_B=budget$curve,curve_conditional_B=budget$curve_conditional,
    double_outer=budget$outer,double_inner=budget$inner,
    result_scope=if(mode=="quick") "reduced-budget executable practice; not book numerical results"
      else "book-budget synthetic practice; runtime/R version may affect results"),mode_path)
  # Generating is cheap and deterministic; never silently load a reader-edited CSV.
  simulation <- simulate_case(3200L,20261010L)
  d <- simulation$data; model <- fit_case(d); identity <- model_identity(model)
  source_files <- file.path(getOption("handbook.repo",course_source_repo),"R",
    c("core.R","interface.R","calibration.R","verify.R","development_intervals.R",
      "competing_branch.R","course.R"))
  if(!all(file.exists(source_files))) stop("Missing R source files for course cache identity")
  source_hashes <- setNames(unname(tools::md5sum(source_files)),basename(source_files))
  ctx <- list(dir=output_dir,mode=mode,budget=budget,d=d,model=model,
              p=predict_case(model,d),identity=identity,
              signature=list(mode=mode,budget=budget,identity=identity,
                data_hash=course_data_hash(d,file.path(output_dir,"cache")),
                R_version=R.version.string,source_hashes=source_hashes))
  apparent <- course_performance(ctx$d$y,ctx$p)
  course_write_csv(data.frame(metric=names(apparent),estimate=sprintf("%.17g",apparent)),
    file.path(output_dir,"results","apparent_metrics.csv"))
  course_write_csv(d,file.path(output_dir,"data","main_case.csv"))
  saveRDS(model,file.path(output_dir,"results","model.rds"),version=3)
  course_write_csv(data.frame(term=names(model$fit$coefficients),
    coefficient=sprintf("%.17g",model$fit$coefficients)),
    file.path(output_dir,"results","model_coefficients.csv"))
  course_write_csv(data.frame(id=d$id,y=d$y,prediction=sprintf("%.17g",ctx$p)),
    file.path(output_dir,"results","predictions.csv"))
  course_write_text(identity,file.path(output_dir,"results","prediction_identity.txt"))
  ctx
}

course_cached <- function(ctx,key,produce) {
  path <- file.path(ctx$dir,"cache",paste0(key,".rds"))
  old <- if(file.exists(path)) readRDS(path) else NULL
  if(!is.null(old) && identical(old$signature,ctx$signature)) return(old$value)
  value <- produce()
  saveRDS(list(signature=ctx$signature,value=value),path,version=3)
  value
}

course_internal <- function(ctx) {
  course_performance(ctx$d$y,ctx$p)
  result <- course_cached(ctx,"internal",function() {
    optimism <- bootstrap_optimism(ctx$d,ctx$model,B=ctx$budget$development)
    conditional <- fixed_model_intervals(ctx$d$y,ctx$p,B=ctx$budget$conditional)
    development <- development_intervals(ctx$d,B_outer=ctx$budget$outer,
      B_inner=ctx$budget$inner)
    list(optimism=optimism,conditional=conditional,development=development)
  })
  if(any(result$conditional$valid_replicates!=ctx$budget$conditional) ||
     any(!is.finite(as.matrix(result$conditional[c("estimate","lower","upper")]))))
    stop("Fixed-model conditional evaluation has failed replicates or unavailable ranges")
  out <- file.path(ctx$dir,"results")
  course_write_csv(result$optimism$table,file.path(out,"bootstrap_optimism.csv"))
  course_write_csv(as.data.frame(result$optimism$replicates),file.path(out,"bootstrap_replicates.csv"))
  course_write_csv(result$conditional,file.path(out,"fixed_model_intervals.csv"))
  course_write_csv(result$development$table,file.path(out,"development_intervals.csv"))
  course_write_csv(result$development$replicates,file.path(out,"development_interval_replicates.csv"))
  course_write_csv(result$optimism$dca,file.path(out,"decision_curve.csv"))
  result
}

course_calibration <- function(ctx) {
  cal <- course_cached(ctx,"calibration",function()
    calibration_artifacts(ctx$d,ctx$model,B=ctx$budget$curve,
                          conditional_B=ctx$budget$curve_conditional))
  out <- file.path(ctx$dir,"results")
  for(name in c("curve","groups","recipe","log"))
    course_write_csv(cal[[name]],file.path(out,paste0("calibration_",name,".csv")))
  if(!isTRUE(cal$success)) stop("Calibration failed; inspect calibration_log.csv")
  plot_calibration(cal,file.path(out,"calibration.pdf"),"pdf")
  plot_calibration(cal,file.path(out,"calibration.png"),"png")
  cal
}

# A deliberately small base-R ridge candidate demonstrates honest comparison
# and nesting. It does not represent random forest, boosting, or TabPFN.
course_ridge_fit <- function(X,y,lambda=10) {
  if(!is.matrix(X) || !is.numeric(X) || nrow(X)<2L || ncol(X)<2L ||
     any(!is.finite(X)) || any(X[,1]!=1))
    stop("Ridge requires a finite numeric design with leading intercept")
  if((!is.numeric(y) && !is.logical(y)) || length(y)!=nrow(X) || anyNA(y) ||
     any(!y %in% 0:1) || length(unique(y))!=2L)
    stop("Ridge requires two observed binary classes")
  if(length(lambda)!=1L || !is.numeric(lambda) || !is.finite(lambda) || lambda<=0)
    stop("Ridge lambda must be one finite positive number; use checked ML for zero penalty")
  beta <- rep(0,ncol(X)); beta[1] <- qlogis(mean(y))
  penalty <- c(0,rep(lambda,ncol(X)-1L))
  objective <- function(b) {
    eta <- as.vector(X%*%b)
    sum(pmax(eta,0)+log1p(exp(-abs(eta)))-y*eta)+sum(penalty*b*b)/2
  }
  converged <- FALSE
  for(iter in seq_len(100L)) {
    p <- plogis(as.vector(X%*%beta)); w <- p*(1-p)
    score <- as.vector(crossprod(X,p-y))+penalty*beta
    if(max(abs(score))/nrow(X)<1e-8) {converged <- TRUE; break}
    H <- crossprod(X,X*w)+diag(penalty,ncol(X))
    step <- as.vector(solve(H,score)); rate <- 1
    while(objective(beta-rate*step)>objective(beta) && rate>2^-20) rate <- rate/2
    beta <- beta-rate*step
  }
  if(!converged || any(!is.finite(beta))) stop("Ridge fit did not converge")
  names(beta) <- colnames(X)
  list(coefficients=beta,lambda=lambda,iterations=iter,converged=converged)
}

course_nested <- function(ctx,tune=FALSE) {
  course_cached(ctx,if(tune) "nested_tuning" else "outer_comparison",function() {
    d <- ctx$d; set.seed(if(tune) 20261032L else 20261031L)
    fold <- sample(rep(1:3,length.out=nrow(d)))
    result <- list(); assignments <- list(); tuning <- list()
    for(k in 1:3) {
      train <- which(fold!=k); test <- which(fold==k)
      # The design knots/scales are prespecified; fitting still uses outer train only.
      Xtr <- make_design(d[train,]); Xte <- make_design(d[test,])
      plain <- fit_case(d[train,]); fixed_lambda <- 10
      if(tune) {
        set.seed(20261040L+k); inner <- sample(rep(1:3,length.out=length(train)))
        candidates <- c(1,10,100); losses <- matrix(NA_real_,3,3)
        for(a in seq_along(candidates)) for(j in 1:3) {
          inner_fit <- course_ridge_fit(Xtr[inner!=j,,drop=FALSE],d$y[train][inner!=j],candidates[a])
          ip <- plogis(as.vector(Xtr[inner==j,,drop=FALSE]%*%inner_fit$coefficients))
          losses[a,j] <- mean((ip-d$y[train][inner==j])^2)
        }
        fixed_lambda <- candidates[which.min(rowMeans(losses))]
        tuning[[k]] <- data.frame(outer_fold=k,lambda=candidates,
          inner_mean_Brier=rowMeans(losses),selected=candidates==fixed_lambda,
          training_n=length(train),test_rows_used_for_tuning=0L)
      }
      ridge <- course_ridge_fit(Xtr,d$y[train],fixed_lambda)
      probabilities <- list(unpenalized_Logistic=predict_case(plain,d[test,]),
        base_R_ridge=plogis(as.vector(Xte%*%ridge$coefficients)))
      for(name in names(probabilities)) {
        evaluation <- course_performance(d$y[test],probabilities[[name]])
        result[[length(result)+1L]] <- data.frame(
          fold=k,candidate=name,lambda=if(name=="base_R_ridge") fixed_lambda else 0,
          train_n=length(train),test_n=length(test),metric=names(evaluation),
          estimate=unname(evaluation),
          scope=if(tune) "outer evaluation after inner tuning" else "outer evaluation; fixed candidate")
      }
      assignments[[k]] <- data.frame(id=d$id,outer_fold=k,
        role=ifelse(seq_len(nrow(d)) %in% train,"TRAIN","TEST"),baseline_training_n=plain$metadata$n)
    }
    list(metrics=do.call(rbind,result),roles=do.call(rbind,assignments),
      tuning=if(tune) do.call(rbind,tuning) else NULL)
  })
}

course_external <- function(ctx) {
  course_cached(ctx,"external",function() {
    # A separate generator seed and event_rate=.18 shift the outcome intercept.
    # This is a synthetic transport exercise, not an actual external cohort.
    ext <- simulate_case(1600L,20261021L,event_rate=.18)$data
    ext$id <- sprintf("V%05d",seq_len(nrow(ext)))
    p <- predict_case(ctx$model,ext); lp <- qlogis(p)
    set.seed(20261022L); update_rows <- sample.int(nrow(ext),800L)
    test_rows <- setdiff(seq_len(nrow(ext)),update_rows)
    intercept <- checked_binomial_fit(matrix(1,length(update_rows),1),
       ext$y[update_rows],offset=lp[update_rows])
    slope <- checked_binomial_fit(cbind(1,lp[update_rows]),ext$y[update_rows])
    tested <- list(frozen=p[test_rows],
      intercept_update=plogis(lp[test_rows]+intercept$coefficients[1]),
      intercept_slope_update=plogis(slope$coefficients[1]+slope$coefficients[2]*lp[test_rows]))
    rows <- lapply(names(tested),function(name) {
      evaluation <- course_performance(ext$y[test_rows],tested[[name]])
      data.frame(rule=name,evaluation_n=length(test_rows),metric=names(evaluation),
        estimate=unname(evaluation),
        scope="independent synthetic held-out evaluation; updates fitted on other 800 rows")
    })
    list(data=ext,metrics=do.call(rbind,rows),assignments=data.frame(id=ext$id,
      role=ifelse(seq_len(nrow(ext)) %in% update_rows,"UPDATE","TEST")),
      updates=data.frame(rule=c("intercept_update","intercept_slope_update"),
        intercept=c(intercept$coefficients[1],slope$coefficients[1]),
        slope=c(1,slope$coefficients[2]),training_n=length(update_rows)))
  })
}

course_run_chapter_impl <- function(chapter,output_dir,mode) {
  ctx <- course_context(output_dir,mode); d <- ctx$d; model <- ctx$model
  here <- file.path(ctx$dir,"chapters",sprintf("ch%02d",chapter))
  dir.create(here,showWarnings=FALSE)
  csv <- function(x,name) course_write_csv(x,file.path(here,name))
  text <- function(x,name) course_write_text(x,file.path(here,name))
  scope <- c(paste0("# 第 ",chapter," 章：",course_titles[chapter]),"",
    paste0("运行模式：",mode,"。全部数据为合成教学数据，模型不供临床决策。"),
    "共享数据与拟合前置条件自动生成；本章产物位于当前目录。",
    "quick 使用较小重抽样预算，数值不能作为书中标准预算结果。",
    "人工审核始终待完成；执行成功不等于临床可用。","")
  instructions <- character()
  switch(as.character(chapter),
    "1"={
      csv(data.frame(check=c("14_inputs","18_parameters","known_two_classes","chronology"),
        observed=c(length(case_recipe()$predictors),ncol(make_design(d))-1L,length(unique(d$y)),
          all(d$collection_time<=d$result_available_time & d$result_available_time<=d$decision_time)),
        status="PASS"),"orientation_checks.csv")
      instructions <- c("1. 查看 mode.csv 和 orientation_checks.csv。",
        "2. 写下病例 A 的结局、决策时点、时间原点以及参数数目。",
        "3. 将合成数据、运行验证和临床验证区分开。")
    },
    "2"={
      text(capture.output(utils::sessionInfo()),"session_info.txt")
      csv(data.frame(item=c("R_version","seed","model_identity","mode"),
        value=c(R.version.string,"20261010",ctx$identity,mode)),"run_record.csv")
      csv(data.frame(package=c("stats","splines","survival"),installed=vapply(
        c("stats","splines","survival"),requireNamespace,logical(1),quietly=TRUE)),"environment.csv")
      instructions <- c("1. 检查 session_info.txt 与 environment.csv。",
        "2. 从项目根目录或任意工作目录运行同一 CLI，比较生成记录。")
    },
    "3"={
      csv(data.frame(n=nrow(d),events=sum(d$y),deaths=sum(d$cd_grade==5),
        missing_input_cells=sum(is.na(d[case_recipe()$predictors])),duplicate_ids=anyDuplicated(d$id)),"data_summary.csv")
      csv(data.frame(field=names(d),class=vapply(d,function(x) class(x)[1],character(1)),
                     missing=vapply(d,function(x) sum(is.na(x)),integer(1))),"field_profile.csv")
      instructions <- c("1. 打开共享 data/main_case.csv；病例数应为 3200，复合事件数为 384。",
        "2. 按中心汇总事件数；检查 true_risk 只用于教学，不能作为拟合输入。")
    },
    "4"={
      csv(data.frame(element=c("population","decision_time","predictors","outcome","origin","window"),
        specification=c("synthetic preoperative cohort; one record per patient",model$target$decision_time,
        paste(case_recipe()$predictors,collapse=";"),model$target$outcome,model$target$outcome_origin,"30 days")),"six_elements.csv")
      eligible <- !duplicated(d$id) & d$decision_time<d$outcome_origin & d$last_contact>=d$outcome_origin
      csv(data.frame(id=d$id,eligible=eligible,reason=ifelse(eligible,"included","invalid chronology/ID")),"cohort_flow.csv")
      instructions <- c("1. 核对 six_elements.csv 与 cohort_flow.csv。",
        "2. 修改规格表时，先说明是否改变预测目标，再修改队列代码。")
    },
    "5"={
      cut <- max(d$last_contact)+1
      A <- make_state(d,cut,outcome="composite_CD")
      B <- make_state(d,cut,outcome="nonfatal_complication")
      stopifnot(identical(as.integer(A$y),as.integer(d$y)))
      csv(data.frame(id=d$id,A_state=A$state,A_y=A$y,B_state=B$state,B_y=B$y),"labels_A_B.csv")
      fixture <- data.frame(outcome_origin=0,event_time=c(5,NA,NA,NA),
        death_time=c(NA,3,NA,NA),last_contact=c(30,30,30,12))
      csv(cbind(fixture,make_state(fixture,30)),"label_fixtures.csv")
      csv(course_negative("event_after_death",function() {
        bad <- fixture[1,]; bad$death_time <- 3; make_state(bad,30)
      }),"negative_checks.csv")
      instructions <- c("1. 比较同一死亡病例的 A_y 与 B_state。",
        "2. 查看随访 12 天且未记录事件的病例；其标签必须未知，不能补为 0。")
    },
    "6"={
      availability <- function(x) {
        required <- c("collection_time","result_available_time","decision_time")
        if(anyNA(x[required])) stop("Unknown result availability")
        if(any(x$collection_time>x$result_available_time |
          x$result_available_time>x$decision_time)) stop("Input unavailable at decision time")
        TRUE
      }
      stopifnot(availability(d))
      csv(data.frame(field=case_recipe()$predictors,available_at_decision=TRUE,
        role="predictor; prespecified transformation"),"variable_review.csv")
      checks <- rbind(course_negative("delayed_result",function() {
        bad <- d[1,]; bad$result_available_time <- bad$decision_time+1; availability(bad)
      }),course_negative("unknown_availability",function() {
        bad <- d[1,]; bad$result_available_time <- NA; availability(bad)
      }))
      csv(checks,"negative_checks.csv")
      instructions <- c("1. 阅读 variable_review.csv：采集时间早于决策仍不足以证明可用。",
        "2. 验证延迟出报告和缺失可得时间都被拒绝。")
    },
    "7"={
      csv(data.frame(input_variables=14L,estimated_parameters=18L,events=sum(d$y),
        descriptive_events_per_parameter=sum(d$y)/18,
        caution="EPP is descriptive; not a sample-size sufficiency criterion"),"parameter_budget.csv")
      set.seed(20261023L); train <- sample.int(nrow(d),2240L); test <- setdiff(seq_len(nrow(d)),train)
      miss <- d; miss$hb[sample.int(nrow(d),480L)] <- NA
      median_train <- median(miss$hb[train],na.rm=TRUE)
      changed_test <- miss; changed_test$hb[test] <- 170
      frozen_again <- median(changed_test$hb[train],na.rm=TRUE)
      stopifnot(identical(median_train,frozen_again))
      imputed <- miss; imputed$hb[is.na(imputed$hb)] <- median_train
      fitted <- fit_case(imputed[train,]); performance_test <- course_performance(imputed$y[test],predict_case(fitted,imputed[test,]))
      csv(data.frame(field="hb",training_n=length(train),training_median=median_train,
        training_missing=sum(is.na(miss$hb[train])),test_missing=sum(is.na(miss$hb[test])),
        method="training-only median; demonstration, NOT multiple imputation"),"missing_recipe.csv")
      csv(data.frame(metric=names(performance_test),estimate=unname(performance_test)),"held_out_median_demo.csv")
      csv(data.frame(id=d$id,role=ifelse(seq_len(nrow(d)) %in% train,"TRAIN","TEST"),
        hb_missing=is.na(miss$hb),hb_after=imputed$hb),"missing_assignments.csv")
      csv(data.frame(check="change_test_inputs_only",training_median=median_train,
        training_median_after_test_change=frozen_again,
        full_data_median_before=median(miss$hb,na.rm=TRUE),
        full_data_median_after=median(changed_test$hb,na.rm=TRUE),
        status="PASS_TRAINING_RECIPE_UNCHANGED"),"boundary_checks.csv")
      instructions <- c("1. 在同一病例框架中确认 14 个变量对应 18 个参数。",
        "2. 训练中位数只用训练行估计，再用于测试行。",
        "3. 此示例不是 MI，不能代替书中的 MICE 流程或其不确定性处理；完整 MI 未由默认命令运行。")
    },
    "8"={
      csv(data.frame(term=names(model$fit$coefficients),coefficient=sprintf("%.17g",model$fit$coefficients)),"coefficients.csv")
      alt <- d[1:10,]; alt$hb <- alt$hb/10; alt$alb <- alt$alb/10; alt$crea <- alt$crea/88.4
      units <- case_units(); units[c("hb","alb","crea")] <- c("g/dL","g/dL","mg/dL")
      altp <- predict_case_with_units(model,alt,units,ctx$identity)
      stopifnot(max(abs(altp-ctx$p[1:10]))<1e-12)
      csv(data.frame(id=alt$id,canonical_prediction=ctx$p[1:10],converted_prediction=altp,
                     absolute_difference=abs(altp-ctx$p[1:10])),"unit_roundtrip.csv")
      csv(data.frame(converged=model$fit$converged,rank=model$fit$rank,
        gradient=model$metadata$gradient,parameters=model$metadata$parameters,
        identity=ctx$identity),"fit_state.csv")
      instructions <- c("1. 核对拟合状态和 19 行含截距系数。",
        "2. 明确声明单位后，g/dL 与 mg/dL 输入应产生相同预测。",
        "3. 原始 predict_case 默认标准单位；界面检查不能凭数值猜测单位。")
    },
    "9"={
      branch <- run_competing_branch(companion_dir=ctx$dir)
      csv(branch$counts,"branch_counts.csv"); csv(branch$coefficients,"branch_coefficients.csv")
      instructions <- c("1. 查看 results/branch_B_counts.csv 与分支运行日志。",
        "2. B 以术后 30 天内首次非致死 III/IV 并发症为目标，死亡是竞争事件。",
        "3. 这是 3 参数关联教学分支，未完成外部验证或临床因果识别。")
    },
    "10"={
      outer <- course_nested(ctx,FALSE)
      csv(outer$metrics,"outer_comparison.csv"); csv(outer$roles,"outer_roles.csv")
      instructions <- c("1. 比较每个外层测试折中的普通 Logistic 与预设 lambda=10 的 ridge。",
        "2. 查看 outer_roles.csv：普通 Logistic 的系数也必须在每个外层训练折重新估计。",
        "3. 这是实际运行的 base R 候选示例，不是书中森林、boosting 或 TabPFN 数字的复现。")
    },
    "11"={
      nested <- course_nested(ctx,TRUE)
      csv(nested$metrics,"nested_metrics.csv"); csv(nested$roles,"outer_roles.csv")
      csv(nested$tuning,"inner_tuning.csv")
      instructions <- c("1. 仅在每个外层训练集内部进行 3 折候选 lambda 调优。",
        "2. 检查 inner_tuning.csv 的 test_rows_used_for_tuning 必须为 0。",
        "3. 选择后用完整外层训练折重拟合，再评价外层测试折。")
    },
    "12"={
      internal <- course_internal(ctx)
      csv(internal$optimism$table,"optimism_corrected_metrics.csv")
      csv(internal$development$table,"development_percentile_ranges.csv")
      csv(internal$conditional,"fixed_apparent_ranges.csv")
      instructions <- c("1. 每个开发 bootstrap 都重新构建设计并重新拟合预先规定的模型。",
        "2. development_percentile_ranges 针对内部校正开发统计量，有限内外预算造成 MC 误差。",
        "3. fixed_apparent_ranges 针对固定模型在原开发样本上的表观评价，不能当作校正结果或外部性能的置信区间。")
    },
    "13"={
      internal <- course_internal(ctx); cal <- course_calibration(ctx)
      csv(cal$curve,"calibration_curve.csv"); csv(cal$groups,"risk_groups.csv")
      pdf(file.path(here,"decision_curve.pdf"),width=7,height=5)
      curve <- internal$optimism$dca
      matplot(curve$threshold,as.matrix(curve[c("corrected_NB","treat_all","treat_none")]),
        type="l",lty=c(1,2,3),col=c("#3679B0","#B6531A","grey50"),
        xlab="Threshold probability",ylab="Net benefit")
      legend("topright",c("Bootstrap corrected","Treat all","Treat none"),
        lty=1:3,col=c("#3679B0","#B6531A","grey50"),bty="n"); dev.off()
      instructions <- c("1. 打开 results/calibration.png，核对平滑曲线与风险分布。",
        "2. 蓝色带是固定模型表观曲线的逐点条件经验范围，橙色校正曲线没有区间带。",
        "3. 阅读 decision_curve.csv：净获益比较同时列出全处理与不处理。")
    },
    "14"={
      external <- course_external(ctx)
      csv(external$data,"synthetic_validation_data.csv"); csv(external$metrics,"held_out_update_evaluation.csv")
      csv(external$assignments,"validation_roles.csv"); csv(external$updates,"update_coefficients.csv")
      instructions <- c("1. 独立种子生成 1600 例、目标事件率 0.18 的合成验证队列，形成截距位移。",
        "2. 其中 800 行估计更新系数，另外 800 行同时评价冻结模型和两种更新。",
        "3. 这是教学迁移实验，不能称为实际外部验证。")
    },
    "15"={
      subgroup <- subgroup_performance(d,ctx$p,B=ctx$budget$subgroup)
      csv(subgroup,"descriptive_subgroups.csv")
      perturbed <- d; perturbed$alb <- pmax(20,d$alb-2)
      pp <- predict_case(model,perturbed)
      csv(data.frame(id=d$id,original=ctx$p,albumin_minus_2=pp,difference=pp-ctx$p),"input_perturbation.csv")
      csv(data.frame(perturbation="albumin minus 2 g/L; lower truncation at 20",mean_change=mean(pp-ctx$p),
        max_absolute_change=max(abs(pp-ctx$p)),scope="prediction sensitivity, not a causal treatment effect"),"sensitivity_summary.csv")
      instructions <- c("1. 亚组指标为开发样本内描述，不能据此宣称某组更优或公平。",
        "2. quick 亚组只有 40 次重复，未达到函数预设的 100 次有效重复要求，区间因此为 NA。",
        "3. 输入扰动展示模型数值敏感性，不代表干预降低白蛋白的因果效应。")
    },
    "16"={
      checks <- rbind(course_negative("unknown_unit",function() {
        units <- case_units(); units["hb"] <- "unknown"; predict_case_with_units(model,d[1,],units)
      }),course_negative("prediction_recipe_changed",function() {
        altered <- model; altered$recipe$age_knots[2] <- 35; assert_model_identity(altered,ctx$identity)
      }),course_negative("complete_separation",function() {
        x <- seq(-2,2,length.out=100); checked_binomial_fit(cbind(1,x),as.integer(x>0))
      }),course_negative("future_label_at_early_cut",function() {
        fixture <- data.frame(outcome_origin=0,event_time=20,death_time=NA_real_,last_contact=30)
        label <- make_state(fixture,10); if(is.na(label$y)) stop("Unknown label cannot be used as event/non-event")
      }))
      stopifnot(all(checks$status=="PASS_EXPECTED_REJECTION")); csv(checks,"negative_checks.csv")
      instructions <- c("1. 阅读 observed_error，四个反例都应触发预期拒绝。",
        "2. 单纯命令执行没有报错不足以验证无效输入被拒绝；检查状态应为 PASS_EXPECTED_REJECTION。")
    },
    "17"={
      explicit <- predict_case_with_units(model,d[1:20,],case_units(),ctx$identity)
      csv(data.frame(id=d$id[1:20],probability=sprintf("%.17g",explicit),
        prediction_identity=ctx$identity,rule_version=model$metadata$version),"example_predictions.csv")
      csv(data.frame(field=names(case_units()),unit=unname(case_units())),"input_units.csv")
      text(c("# 教学模型登记",paste0("完整预测规则身份：",ctx$identity),
        "MD5 只检测非恶意一致性，不是数字签名。",
        "状态：SHADOW_TEACHING_ONLY；人工审核：PENDING。",
        "模型未进行临床部署授权。"),"model_registry.md")
      instructions <- c("1. 用单位声明和完整身份调用预测接口，查看 example_predictions.csv。",
        "2. 改动任一年龄结点后应触发身份不一致；系数相同不等于预测规则相同。")
    },
    "18"={
      internal <- course_internal(ctx); cal <- course_calibration(ctx); external <- course_external(ctx)
      metrics <- internal$optimism$table
      report <- c("# 合成教学案例运行报告","",paste0("模式：",mode),
        paste0("数据：",nrow(d)," 例；复合事件 ",sum(d$y),"；14 输入，18 参数。"),
        paste0("预测规则身份：",ctx$identity),"",
        "|指标|表观值|内部校正值|乐观度 MCSE|","|---|---:|---:|---:|",
        apply(metrics,1,function(row) paste0("|",paste(row[c("metric","apparent","corrected","optimism_MCSE")],collapse="|"),"|")),
        "","![平滑校准与风险分布](../../results/calibration.png)","",
        "重抽样范围的目标见本目录 interval_targets.csv。曲线带为固定模型表观曲线逐点条件范围；校正曲线只有点估计。",
        "独立合成队列更新实验见本目录 synthetic_update_evaluation.csv；更新估计和评价使用不同病例。",
        "本运行没有临床外部验证、完整 MICE 链、AI 人工签名或临床部署授权。")
      text(report,"report.md")
      csv(data.frame(artifact=c("development_intervals.csv","fixed_model_intervals.csv","calibration_curve.csv"),
        target=c("internally corrected development statistic","fixed model apparent performance",
                 "fixed model apparent smooth curve, pointwise"),
        excludes=c("fixed final model external performance","optimism-corrected performance",
                   "corrected curve interval and simultaneous coverage")),"interval_targets.csv")
      csv(external$metrics,"synthetic_update_evaluation.csv")
      instructions <- c("1. 阅读 report.md，并将重抽样目标与每张表逐一对应。",
        "2. 练习撰写限制；不把运行成功、教学外部样本或人工待审写成临床已验证。")
    },
    "19"={
      checklist <- data.frame(item=c("target","cohort_flow","complete_rule","validation_scope",
        "calibration_curve","risk_distribution","DCA","human_review","ethics_and_real_data"),
        evidence=c("ch04/six_elements.csv","ch04/cohort_flow.csv","results/model.rds + identity",
          "ch12 + ch14 methods","results/calibration.png","calibration_groups.csv",
          "results/decision_curve.csv","reader must supply","not applicable to synthetic example"),
        status=c(rep("READER_VERIFY",7),"PENDING","SYNTHETIC_ONLY"))
      csv(checklist,"submission_checklist.csv")
      text(c("# 投稿材料练习","", "请读者根据实际研究补充以下内容：",
        "- 队列与六要素、纳入排除、缺失处理及其验证边界。",
        "- 完整模型规则、验证目标、校准图、风险分布及决策曲线。",
        "- 样本量依据、限制、伦理审批、数据与代码可得性。",
        "作者、机构、伦理编号、杂志要求及人工签名不得从教学案例推定。",
        "本命令只产生本地材料，没有提交稿件或联系第三方。"),"submission_materials.md")
      instructions <- c("1. 检查清单中的每一项 READER_VERIFY，填写真实证据位置。",
        "2. 读取者需要实际执行相应章节后再整理材料，清单生成不是完成审阅。")
    },
    "20"={
      csv(data.frame(action=c("generated_synthetic_lesson","reviewed_real_data_or_clinical_claims",
          "human_signed_off"),status=c("RECORDED","NOT_PERFORMED","PENDING"),
        actor=c("course CLI","reader","human reviewer required"),mode=mode),"ai_audit.csv")
      csv(data.frame(role=c("clinical_reviewer","statistical_reviewer","implementation_reviewer"),
        name=NA_character_,decision="PENDING",date=NA_character_),"human_review.csv")
      text(c("# 人工审核练习","", "核对所有命令、原始产物、错误日志与方法边界。",
        "真实数据与临床解释需要人工审核，模型不能替代签字者。",
        "生成记录说明代码做了什么；它不能证明作者本人已阅读或批准。"),"audit_instructions.md")
      instructions <- c("1. 对照 ai_audit.csv 区分已执行与未执行行为。",
        "2. 人工审阅完成前保留 human_review.csv 的 PENDING；不要自动生成姓名、日期或签名。")
    })
  text(c(scope,"## 操作与核对",instructions),"lesson.md")
  artifacts <- list.files(here,recursive=TRUE,full.names=TRUE)
  course_write_status(chapter,ctx$dir,mode,"PASS_EXECUTED")
  cat(sprintf("CHAPTER_%02d_PASS mode=%s output=%s\n",chapter,mode,here))
  invisible(list(status="PASS_EXECUTED",chapter=chapter,mode=mode,
                 artifacts=c(artifacts,file.path(here,"status.csv"))))
}

run_chapter <- function(chapter,output_dir,mode=c("quick","book")) {
  mode <- match.arg(mode)
  chapter <- course_chapter_number(chapter)
  # A conflicting mode must not overwrite any existing chapter status.
  course_validate_mode(output_dir,mode)
  dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
  output_dir <- normalizePath(output_dir,winslash="/",mustWork=TRUE)
  course_validate_mode(output_dir,mode)
  course_write_status(chapter,output_dir,mode,"RUNNING")
  tryCatch(course_run_chapter_impl(chapter,output_dir,mode),error=function(e) {
    course_write_status(chapter,output_dir,mode,"FAIL",conditionMessage(e))
    stop(e)
  },interrupt=function(e) {
    course_write_status(chapter,output_dir,mode,"FAIL","Run interrupted before successful completion")
    stop(e)
  })
}
