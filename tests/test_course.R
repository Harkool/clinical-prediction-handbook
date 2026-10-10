# Rscript --vanilla tests/test_course.R [absolute output directory]
if(.Platform$OS.type=="windows")
  invisible(Sys.setlocale("LC_CTYPE","English_United States.utf8"))
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE))
repo <- dirname(dirname(normalizePath(script,winslash="/",mustWork=TRUE)))
options(handbook.repo=repo)
for(module in c("core","interface","calibration","verify","development_intervals","competing_branch","course"))
  source(file.path(repo,"R",paste0(module,".R")),encoding="UTF-8")
args <- commandArgs(TRUE)
output <- if(length(args)) args[1] else file.path(repo,"outputs","test-course")
verify_logic()
for(ch in 1:20) run_chapter(ch,output,"quick")
read <- function(ch,file) read.csv(file.path(output,"chapters",sprintf("ch%02d",ch),file),stringsAsFactors=FALSE)
checks <- list()
record <- function(name,pass,scope) {
  if(!isTRUE(pass)) stop("Course test failed: ",name)
  checks[[length(checks)+1L]] <<- data.frame(check=name,status="PASS",scope=scope)
}
manifest <- read.csv(file.path(output,"manifest.csv"))
record("20_chapters_executed",identical(as.integer(manifest$chapter),1:20),"quick actual runs")
summary <- read(3,"data_summary.csv")
record("cohort_reference_counts",summary$n==3200 && summary$events==384 && summary$deaths==39,"synthetic generator")
labels <- read(5,"labels_A_B.csv")
record("unknown_is_not_zero",is.na(read(5,"label_fixtures.csv")$y[4]),"follow-up fixture")
record("death_target_differs",sum(labels$A_y==1)==384 && sum(labels$B_state=="competing")==39,"A/B target")
record("availability_rejected",all(read(6,"negative_checks.csv")$status=="PASS_EXPECTED_REJECTION"),"late/unknown results")
boundary <- read(7,"boundary_checks.csv")
record("missing_recipe_excludes_test",boundary$training_median==boundary$training_median_after_test_change &&
  boundary$full_data_median_before!=boundary$full_data_median_after,"change held-out inputs only")
record("units_prediction_equal",max(read(8,"unit_roundtrip.csv")$absolute_difference)<1e-12,"explicit conversions")
record("outer_baseline_refitted",all(read(10,"outer_roles.csv")$baseline_training_n<3200),"no full-cohort coefficient reuse")
record("tuning_excludes_outer_test",all(read(11,"inner_tuning.csv")$test_rows_used_for_tuning==0),"nested actual code")
record("range_targets_separate",length(unique(read(18,"interval_targets.csv")$target))==3,"three interval estimands")
record("negative_fixtures_rejected",all(read(16,"negative_checks.csv")$status=="PASS_EXPECTED_REJECTION"),"units/rule/separation/future label")
record("synthetic_update_roles_disjoint",sum(read(14,"validation_roles.csv")$role=="UPDATE")==800 &&
  sum(read(14,"validation_roles.csv")$role=="TEST")==800,"synthetic update split")
record("validation_ids_distinct",!any(read(14,"validation_roles.csv")$id %in%
  read.csv(file.path(output,"data","main_case.csv"))$id),"V IDs distinct from development A IDs")
record("human_not_fabricated",all(read(20,"human_review.csv")$decision=="PENDING") &&
  all(is.na(read(20,"human_review.csv")$name)),"unperformed human review")
curve <- read.csv(file.path(output,"results","calibration_curve.csv"))
record("calibration_all_replicates",all(curve$bootstrap_success==20) && all(curve$conditional_success==40),"quick curve budgets")
identity <- readLines(file.path(output,"results","prediction_identity.txt"))
model <- readRDS(file.path(output,"results","model.rds"))
record("complete_identity_valid",identical(model_identity(model),identity),"canonical prediction rule")
Xbad <- cbind(1,seq(-1,1,length.out=100))
xsep <- seq(-2,2,length.out=100)
invalid <- rbind(
  course_negative("ridge_nonbinary_labels",function()
    course_ridge_fit(Xbad,seq(.2,.8,length.out=100),10)),
  course_negative("ridge_negative_lambda",function()
    course_ridge_fit(Xbad,rep(0:1,50),-1)),
  course_negative("ridge_zero_penalty_separation",function()
    course_ridge_fit(cbind(1,xsep),as.integer(xsep>0),0)),
  course_negative("undefined_calibration_metric",function()
    course_performance(rep(0:1,50),rep(.5,100))))
course_write_csv(invalid,file.path(output,"invalid_input_checks.csv"))
record("ridge_and_metric_invalid_inputs_rejected",
  all(invalid$status=="PASS_EXPECTED_REJECTION"),"actual invalid-y/lambda/constant-p fixtures")
before <- read(8,"status.csv")
original_fit_case <- fit_case
failure <- tryCatch({
  fit_case <- function(...) stop("SIMULATED_TEST_FIT_FAILURE")
  tryCatch({run_chapter(8,output,"quick"); "unexpected success"},
           error=function(e) conditionMessage(e))
},finally={fit_case <- original_fit_case})
after <- read(8,"status.csv")
failed_manifest <- read.csv(file.path(output,"manifest.csv"),stringsAsFactors=FALSE)
failed_row <- failed_manifest[failed_manifest$chapter==8,,drop=FALSE]
course_write_csv(data.frame(chapter=8,before_status=before$status,
  after_status=after$status,manifest_status=failed_row$status,
  observed_error=failure,status_error=after$scope,
  manifest_error=failed_row$scope,
  failure_file=readLines(file.path(output,"chapters","ch08","failure.txt"))),
  file.path(output,"failure_status_check.csv"))
record("failed_rerun_invalidates_previous_PASS",before$status=="PASS_EXECUTED" &&
  after$status=="FAIL" && failed_row$status=="FAIL" &&
  failure=="SIMULATED_TEST_FIT_FAILURE" && after$scope==paste0("FAILED: ",failure) &&
  failed_row$scope==paste0("FAILED: ",failure) &&
  identical(readLines(file.path(output,"chapters","ch08","failure.txt")),failure),
  "actual failed rerun after prior chapter success")
snapshot <- readLines(file.path(output,"chapters","ch08","status.csv"))
mode_conflict <- tryCatch({run_chapter(8,output,"book"); FALSE},error=function(e) TRUE)
record("mode_conflict_preserves_existing_status",mode_conflict &&
  identical(snapshot,readLines(file.path(output,"chapters","ch08","status.csv"))),
  "book request cannot overwrite quick status")
run_chapter(8,output,"quick")
record("successful_rerun_recovers_status",read(8,"status.csv")$status=="PASS_EXECUTED",
  "restored original fit function")
course_write_csv(do.call(rbind,checks),file.path(output,"test_summary.csv"))
cat("COURSE_TESTS_PASS checks=",length(checks),"\n",sep="")
