# Explicit units and complete prediction-rule consistency, for teaching case A.
# Source core.R first. MD5 here detects accidental inconsistency, not hostile edits.
case_units <- function() c(age="year",bmi="kg/m2",male="0/1",asa="I-IV",
  emergency="0/1",diabetes="0/1",copd="0/1",smoker="0/1",chf="0/1",
  sbp="mmHg",hr="beat/min",hb="g/L",alb="g/L",crea="umol/L")

case_unit_conversions <- function() data.frame(field=c("hb","alb","crea"),
  from=c("g/dL","g/dL","mg/dL"),to=c("g/L","g/L","umol/L"),
  multiplier=c(10,10,88.4),stringsAsFactors=FALSE)

case_input_ranges <- function() list(age=c(18,90),bmi=c(16,45),
  male=c(0,1),asa=c(1,4),emergency=c(0,1),diabetes=c(0,1),copd=c(0,1),
  smoker=c(0,1),chf=c(0,1),sbp=c(75,200),hr=c(40,140),hb=c(70,180),
  alb=c(20,52),crea=c(30,400))

normalise_case_units <- function(d,units) {
  canonical <- case_units()
  if(!is.character(units) || is.null(names(units)) || anyDuplicated(names(units)) ||
     !setequal(names(units),names(canonical)) || anyNA(units))
    stop("Declare exactly one explicit unit for each of the 14 inputs")
  if(length(setdiff(names(canonical),names(d)))) stop("Missing raw inputs")
  conversions <- case_unit_conversions()
  for(k in names(canonical)) {
    if(!is.numeric(d[[k]]) || any(!is.finite(d[[k]]))) stop("Non-finite/non-numeric input: ",k)
    if(units[[k]] != canonical[[k]]) {
      i <- which(conversions$field==k & conversions$from==units[[k]] &
                 conversions$to==canonical[[k]])
      if(length(i)!=1L) stop("Unsupported explicit unit for ",k,": ",units[[k]])
      d[[k]] <- d[[k]]*conversions$multiplier[i]
    }
  }
  ranges <- case_input_ranges()
  for(k in names(ranges)) if(any(d[[k]]<ranges[[k]][1] | d[[k]]>ranges[[k]][2]))
    stop("Outside prespecified synthetic teaching domain: ",k)
  binary <- c("male","emergency","diabetes","copd","smoker","chf")
  if(any(!as.matrix(d[binary]) %in% 0:1) || any(!d$asa %in% 1:4))
    stop("Invalid binary/ASA level")
  d
}

# Typed, length-prefixed UTF-8 encoding. Named list fields are sorted;
# vector order and names (including coefficient order) are preserved.
canonical_encode <- function(x) {
  atom <- function(s) paste0(nchar(enc2utf8(s),type="bytes"),":",enc2utf8(s))
  nm <- if(is.null(names(x))) "unnamed" else paste0("names",canonical_encode(unname(names(x))))
  if(is.null(x)) return("NULL")
  if(is.list(x)) {
    if(!is.null(names(x))) {
      if(anyDuplicated(names(x))) stop("Duplicate payload fields")
      x <- x[order(names(x),method="radix")]
    }
    return(paste0("list",length(x),"[",paste(vapply(seq_along(x),function(i)
      paste0(if(is.null(names(x))) "" else atom(names(x)[i]),
             atom(canonical_encode(x[[i]]))),character(1)),collapse=""),"]"))
  }
  if(!is.atomic(x) || anyNA(x)) stop("Unsupported/missing canonical payload value")
  values <- if(is.double(x)) vapply(x,function(v) sprintf("%.17g",v),character(1))
    else if(is.logical(x)) ifelse(x,"TRUE","FALSE") else as.character(x)
  paste0(typeof(x),length(x),"[",nm,";",paste(vapply(values,atom,character(1)),collapse=""),"]")
}

canonical_model_payload <- function(model) {
  beta <- model$fit$coefficients
  if(is.null(names(beta)) || any(!is.finite(beta))) stop("Invalid named coefficients")
  source_body <- function(f) paste(deparse(body(f),width.cutoff=500L),collapse="\n")
  spec <- list(schema="case-A-prediction-rule-v1",coefficients=beta,
    recipe=model$recipe,target=model$target,version=model$metadata$version,
    units=case_units(),unit_conversions=as.list(case_unit_conversions()),
    teaching_domain=case_input_ranges(),
    level_definitions=list(binary=c(0L,1L),asa=1:4,asa_reference=1L),
    implementation=list(rcs_age=source_body(rcs_age),make_design=source_body(make_design),
      predict_case=source_body(predict_case),normalise_units=source_body(normalise_case_units)))
  canonical_encode(spec)
}

model_identity <- function(model) {
  file <- tempfile("case-model-identity-")
  on.exit(unlink(file),add=TRUE)
  con <- file(file,"wb"); on.exit(close(con),add=TRUE)
  writeChar(enc2utf8(canonical_model_payload(model)),con,eos=NULL,useBytes=TRUE)
  close(con); on.exit(unlink(file),add=FALSE)
  paste0("md5:",unname(tools::md5sum(file)))
}

assert_model_identity <- function(model,expected) {
  if(length(expected)!=1L || is.na(expected) || !grepl("^md5:[0-9a-f]{32}$",expected))
    stop("Expected identity must declare the MD5 algorithm")
  if(!identical(model_identity(model),expected)) stop("Complete prediction-rule identity mismatch")
  invisible(TRUE)
}

predict_case_with_units <- function(model,d,units,expected_identity=NULL) {
  if(!is.null(expected_identity)) assert_model_identity(model,expected_identity)
  predict_case(model,normalise_case_units(d,units))
}
