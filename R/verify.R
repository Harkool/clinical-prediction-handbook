verify_logic <- function() {
  row <- function(ev=NA_real_,death=NA_real_,last=30)
    data.frame(outcome_origin=0,event_time=ev,death_time=death,last_contact=last)
  # Cut-off day 10 may not use an event on day 20 or a contact on day 40.
  stopifnot(make_state(row(ev=20,last=10),10)$state=="incomplete_admin")
  stopifnot(make_state(row(last=40),10)$state=="incomplete_admin")
  # A chronological impossibility must be rejected, not labelled event.
  failure <- tryCatch({make_state(row(ev=5,death=3),30); FALSE},error=function(e)TRUE)
  stopifnot(failure)
  # Death is a main event for CD, a competitor only for the explicit B target.
  stopifnot(make_state(row(death=3),30,outcome="composite_CD")$y==1L)
  stopifnot(make_state(row(death=3),30,outcome="nonfatal_complication")$state=="competing")
  stopifnot(is.na(make_state(row(last=12),30)$y))
  stopifnot(auc(c(1,0),c(.5,.5))==.5)
  invisible(TRUE)
}
