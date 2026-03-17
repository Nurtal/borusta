library(Boruta)
library(randomForest)

for (dataset in c("iris", "wine")) {
  path <- sprintf("benchmark/%s.csv", dataset)
  df <- read.csv(path)
  X  <- df[, !names(df) %in% "label"]
  y  <- as.factor(df$label)
  n_classes <- length(levels(y))

  cat(sprintf("\n%s\n", strrep("=", 60)))
  cat(sprintf("Dataset: %s  (%d obs × %d features, %d classes)\n",
              dataset, nrow(X), ncol(X), n_classes))

  set.seed(42)
  t0 <- proc.time()
  res <- Boruta(x=X, y=y, maxRuns=100, pValue=0.01, mcAdj=TRUE, doTrace=0)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  dec       <- res$finalDecision
  confirmed <- names(dec[dec == "Confirmed"])
  rejected  <- names(dec[dec == "Rejected"])
  tentative <- names(dec[dec == "Tentative"])
  n_iter    <- length(res$ImpHistory[,1])

  cat(sprintf("=== Boruta R (%d iterations, %.3fs) ===\n", n_iter, elapsed))
  cat(sprintf("  Confirmed  (%2d): %s\n", length(confirmed), paste(confirmed, collapse=", ")))
  cat(sprintf("  Rejected   (%2d): %s\n", length(rejected),  paste(rejected,  collapse=", ")))
  cat(sprintf("  Tentative  (%2d): %s\n", length(tentative), paste(tentative, collapse=", ")))
  cat(sprintf("  Elapsed: %.3fs\n", elapsed))
}
