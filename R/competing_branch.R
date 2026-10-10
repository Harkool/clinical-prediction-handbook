# Documentation checked locally: survival 3.8.6 help("finegray") and its
# installed function source: factor first level is censor, and cluster-based
# robust standard errors are recommended after the weighted Cox expansion.
# Independent competing-risk branch B from the same synthetic teaching cohort.
# Source from companion/: source("R/competing_branch.R"); run_competing_branch()
# Or: Rscript --vanilla /absolute/path/to/companion/R/competing_branch.R
# Target B: first nonfatal CD III/IV by day 30, with CD V death competing.
# This is not case A's CD >= III including-death binary target, its 18-parameter
# model, an update of that model, or a clinically validated deployment model.
if (.Platform$OS.type == "windows")
  invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))

run_competing_branch <- local({
  # Capture the source file when sourced; use --file when executed by Rscript.
  frames <- sys.frames()
  source_files <- unlist(lapply(frames, function(frame)
    if (is.character(frame$ofile)) frame$ofile else NULL), use.names=FALSE)
  arguments <- commandArgs(trailingOnly=FALSE)
  executable_files <- sub("^--file=", "", arguments[grepl("^--file=", arguments)])
  if (length(source_files)) {
    source_path <- normalizePath(tail(source_files, 1), winslash="/", mustWork=TRUE)
    default_dir <- dirname(dirname(source_path))
  } else if (length(executable_files)) {
    source_path <- normalizePath(executable_files[1], winslash="/", mustWork=TRUE)
    default_dir <- dirname(dirname(source_path))
  } else default_dir <- normalizePath(getwd(), winslash="/", mustWork=TRUE)

  function(companion_dir=default_dir) {
    if (.Platform$OS.type == "windows")
      invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
    companion_dir <- normalizePath(companion_dir, winslash="/", mustWork=TRUE)
    input <- file.path(companion_dir, "data", "main_case.csv")
    if (!file.exists(input)) stop("Generate case A first: missing data/main_case.csv")
    if (!requireNamespace("survival", quietly=TRUE))
      stop("The installed survival package is required; no automatic installation.")

    # Keep all branch outputs separate from A. Logging is evidence of this call.
    results_dir <- file.path(companion_dir, "results")
    dir.create(results_dir, recursive=TRUE, showWarnings=FALSE)
    log_path <- file.path(results_dir, "branch_B_run.log")
    log_connection <- file(log_path, open="wt", encoding="UTF-8")
    sink(log_connection, split=TRUE)
    on.exit({sink(); close(log_connection)}, add=TRUE)
    old_digits <- getOption("digits")
    options(digits=17)
    on.exit(options(digits=old_digits), add=TRUE)
    cat("Branch B run:", format(Sys.time(), tz="UTC", usetz=TRUE), "\n")
    cat("Input:", normalizePath(input, winslash="/"), "\n")
    cat("Input MD5:", unname(tools::md5sum(input)), "\n")
    cat("Target: first nonfatal CD III/IV within 30 days from surgery start;",
        "CD V death before that event is competing.\n")
    cat("A targets CD >= III including death; B is a separate 3-parameter",
        "association demonstration, not an A update or deployment model.\n")

    d <- read.csv(input, fileEncoding="UTF-8", stringsAsFactors=FALSE,
                  check.names=FALSE)
    required <- c("id", "age", "bmi", "male", "y", "cd_grade",
                  "decision_time", "outcome_origin", "collection_time",
                  "result_available_time", "event_time", "death_time",
                  "last_contact")
    missing <- setdiff(required, names(d))
    if (length(missing)) stop("Missing fields: ", paste(missing, collapse=", "))
    if (anyNA(d[c("id", "age", "bmi", "male", "y", "cd_grade",
                  "decision_time", "outcome_origin", "collection_time",
                  "result_available_time", "last_contact")]))
      stop("Branch B requires known follow-up and complete prespecified inputs.")
    if (anyDuplicated(d$id) || any(!nzchar(d$id)))
      stop("Each source row must identify one patient.")
    numeric_fields <- setdiff(required, "id")
    if (!all(vapply(d[numeric_fields], is.numeric, logical(1))))
      stop("The interchange timestamps and measurements must be numeric.")
    if (any(!is.finite(as.matrix(d[c("age","bmi","male","outcome_origin",
                                    "last_contact")]))) ||
        any(!d$male %in% 0:1) || any(!d$cd_grade %in% 0:5) ||
        any(!d$y %in% 0:1))
      stop("Non-finite inputs or invalid binary/grade codes.")
    stopifnot(all(d$collection_time <= d$result_available_time),
              all(d$result_available_time <= d$decision_time),
              all(d$decision_time < d$outcome_origin),
              identical(as.integer(d$cd_grade >= 3), as.integer(d$y)))

    # All times in the CSV are numeric days on a common calendar scale.
    # Convert event, death and last-contact times to days AFTER surgery start,
    # never relative to decision_time or to a raw Unix/calendar timestamp.
    nonfatal_day <- d$event_time - d$outcome_origin
    death_day <- d$death_time - d$outcome_origin
    last_day <- d$last_contact - d$outcome_origin
    if (any(last_day <= 0) ||
        any(!is.na(nonfatal_day) & (!is.finite(nonfatal_day) | nonfatal_day <= 0)) ||
        any(!is.na(death_day) & (!is.finite(death_day) | death_day <= 0)))
      stop("Nonpositive or non-finite post-origin outcome/follow-up times.")
    nonfatal_grade <- d$cd_grade %in% c(3L,4L)
    death_grade <- d$cd_grade == 5L
    if (any(nonfatal_grade & is.na(nonfatal_day)) ||
        any(death_grade & is.na(death_day)))
      stop("An adjudicated event grade needs its occurrence timestamp.")
    # The synthetic source stores one final grade: III/IV have nonfatal times,
    # V has a death time. It does not encode earlier III/IV events for CD V.
    # Do not fabricate a nonfatal event time for the fatal patients.
    if (any(!nonfatal_grade & !is.na(nonfatal_day)) ||
        any(!death_grade & !is.na(death_day)))
      stop("Timestamp/grade convention differs from the documented source.")
    followup_day <- pmin(last_day, 30)
    nf <- ifelse(nonfatal_grade, nonfatal_day, Inf)
    dd <- ifelse(death_grade, death_day, Inf)
    code <- integer(nrow(d))
    code[is.finite(nf) & nf <= followup_day & nf < dd] <- 1L
    # Explicit tie convention: death wins; real equal-resolution records need
    # adjudication before use. No tied nonfatal/death records occur in this file.
    code[is.finite(dd) & dd <= followup_day & dd <= nf] <- 2L
    time_day <- followup_day
    time_day[code == 1L] <- nf[code == 1L]
    time_day[code == 2L] <- dd[code == 2L]
    status_levels <- c("censor", "nonfatal_CD_III_IV", "death_CD_V")
    status <- factor(code, levels=0:2, labels=status_levels)
    stopifnot(identical(levels(status)[1], "censor"),
              all(time_day > 0 & time_day <= 30),
              all(last_day >= 30 - 1e-9),
              all(as.integer(code != 0L) == d$y),
              sum(code == 1L) > 0L, sum(code == 2L) > 0L)

    # Zero reference: age 50 years, BMI 25, male=0.
    # Only these THREE prespecified baseline covariates may enter a model.
    whitelist <- c("age10", "bmi5", "male")
    branch <- data.frame(time_day=time_day, status=status,
                         age10=(d$age-50)/10, bmi5=(d$bmi-25)/5,
                         male=d$male, patient_id=d$id)
    if (!all(is.finite(as.matrix(branch[whitelist]))))
      stop("Non-finite design.")
    stopifnot(!any(c("status","patient_id","fgwt","id") %in% whitelist))

    # finegray copies RHS fields to its expanded data. patient_id is carried
    # as metadata ONLY so expanded rows can be clustered by original patient.
    # No regression is fitted here. Never use '~ .' to create a model.
    fg_formula <- survival::Surv(time_day, status) ~
      age10 + bmi5 + male + patient_id
    fg <- survival::finegray(fg_formula, data=branch,
                            etype="nonfatal_CD_III_IV",
                            id=patient_id, na.action=stats::na.fail)
    required_expanded <- c(whitelist, "patient_id", "fgstart", "fgstop",
                          "fgstatus", "fgwt")
    stopifnot(all(required_expanded %in% names(fg)),
              all(fg$fgwt > 0 & is.finite(fg$fgwt)),
              all(fg$fgstop > fg$fgstart),
              sum(fg$fgstatus) == sum(code == 1L),
              all(fg$patient_id %in% branch$patient_id))

    model_formula <- survival::Surv(fgstart, fgstop, fgstatus) ~
      age10 + bmi5 + male
    fit <- survival::coxph(model_formula, data=fg, weights=fgwt,
                           cluster=patient_id, robust=TRUE, ties="breslow",
                           x=TRUE, model=TRUE, singular.ok=FALSE,
                           control=survival::coxph.control(iter.max=100,
                                                         eps=1e-10))
    stopifnot(identical(names(stats::coef(fit)), whitelist),
              identical(colnames(fit$x), whitelist),
              all(is.finite(stats::coef(fit))),
              all(is.finite(stats::vcov(fit))),
              all(diag(stats::vcov(fit)) > 0),
              length(fit$iter) == 1L, fit$iter < 100L,
              abs(fit$loglik[2]) < Inf)

    # basehaz(centered=FALSE) and predict(reference="zero") use the SAME zero
    # covariate reference, avoiding an implicit centered baseline mismatch.
    zero <- data.frame(age10=0, bmi5=0, male=0)
    zero_lp <- as.numeric(stats::predict(fit, newdata=zero,
                                        type="lp", reference="zero"))
    lp <- as.numeric(stats::predict(fit, newdata=branch,
                                   type="lp", reference="zero"))
    manual_lp <- as.vector(as.matrix(branch[whitelist]) %*% stats::coef(fit))
    baseline <- survival::basehaz(fit, centered=FALSE)
    H30 <- if (any(baseline$time <= 30))
      tail(baseline$hazard[baseline$time <= 30],1) else 0
    cif30 <- -expm1(-H30 * exp(lp))
    zero_cif30 <- -expm1(-H30)
    stopifnot(abs(zero_lp) < 1e-12, max(abs(lp-manual_lp)) < 1e-10,
              all(is.finite(cif30)), all(cif30 >= 0 & cif30 <= 1),
              is.finite(zero_cif30), zero_cif30 >= 0, zero_cif30 <= 1)

    aj <- survival::survfit(survival::Surv(time_day,status) ~ 1,
                           data=branch)
    aj_cif30 <- tail(aj$pstate[, "nonfatal_CD_III_IV"],1)
    observed_cif30 <- mean(code == 1L)
    # Complete follow-up means AJ's marginal CIF equals the event proportion.
    stopifnot(abs(aj_cif30-observed_cif30) < 1e-10)

    counts <- data.frame(
      branch="B", target="nonfatal CD III/IV before competing CD V death",
      origin="surgery start", window_days=30,
      state=status_levels, n=as.integer(table(status)),
      proportion=as.numeric(table(status))/nrow(d),
      total_n=nrow(d), source_A_composite_events=sum(d$y),
      model_parameters=length(whitelist), expanded_rows=nrow(fg),
      full_30_day_followup=TRUE, row.names=NULL)
    estimates <- stats::coef(fit)
    robust_se <- sqrt(diag(stats::vcov(fit)))
    coefficients <- data.frame(
      branch="B", term=names(estimates),
      coefficient=unname(estimates), robust_SE=unname(robust_se),
      subdistribution_HR=unname(exp(estimates)),
      HR_lower_95=unname(exp(estimates-1.96*robust_se)),
      HR_upper_95=unname(exp(estimates+1.96*robust_se)),
      definition=c("(age-50)/10", "(bmi-25)/5", "male=1 versus 0"),
      zero_reference="age=50, BMI=25, male=0",
      target="nonfatal CD III/IV, CD V competing",
      model_use="synthetic branch B demonstration; no clinical deployment",
      row.names=NULL)
    write.csv(counts, file.path(results_dir,"branch_B_counts.csv"),
              row.names=FALSE, fileEncoding="UTF-8", na="NA")
    # Preserve complete double precision in the coefficient interchange file.
    export_coefficients <- coefficients
    numeric_columns <- names(coefficients)[vapply(coefficients, is.numeric,
                                                  logical(1))]
    for (field in numeric_columns)
      export_coefficients[[field]] <- sprintf("%.17g", coefficients[[field]])
    coefficient_path <- file.path(results_dir,"branch_B_finegray_coefficients.csv")
    write.csv(export_coefficients, coefficient_path,
              row.names=FALSE, fileEncoding="UTF-8", na="NA")
    roundtrip <- read.csv(coefficient_path, fileEncoding="UTF-8",
                          stringsAsFactors=FALSE)
    stopifnot(identical(roundtrip$term, coefficients$term),
              max(abs(roundtrip$coefficient-coefficients$coefficient)) < 1e-16)
    cat("\nCounts:\n"); print(counts)
    cat("\nCoefficients and robust intervals:\n"); print(coefficients)
    cat("\nChecks: censor first level; whitelist model matrix; ID only cluster;",
        "fgwt only weights; post-origin days; zero LP; bounded CIF; AJ counts: PASS\n")
    cat("Cox iterations:", fit$iter, "; log likelihood:", fit$loglik[2], "\n")
    cat("fgwt range:", range(fg$fgwt), "; expanded rows:", nrow(fg), "\n")
    cat("Zero-reference cumulative subhazard at day30:", H30,
        "; zero-reference CIF:", zero_cif30, "\n")
    cat("Marginal AJ CIF day30:", aj_cif30,
        "; observed nonfatal proportion:", observed_cif30,
        "; mean fitted CIF:", mean(cif30), "\n")
    cat("Baseline: basehaz(centered=FALSE); LP: predict(reference='zero').\n")
    cat("No resampling validation, external validation, PH assessment or",
        "clinical causal identification performed by this branch.\n")
    cat("Synthetic source limitation: CD V contains no earlier III/IV history;",
        "the branch uses only recorded nonfatal events and competing deaths.\n")
    cat("\nEnvironment:\n"); print(utils::sessionInfo())
    cat("\nBRANCH_B_SUCCESS\n")
    invisible(list(target=counts$target[1], counts=counts,
                   coefficients=coefficients, fit=fit,
                   baseline_zero=baseline, checks=list(zero_lp=zero_lp,
                   aj_cif30=aj_cif30, observed_cif30=observed_cif30,
                   mean_fitted_cif30=mean(cif30))))
  }
})
if (sys.nframe() == 0L) run_competing_branch()
