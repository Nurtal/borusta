library(Boruta)
library(randomForest)

# ── Load dataset ──────────────────────────────────────────────────────────────
df <- read.csv("benchmark/dataset.csv")
X  <- df[, !names(df) %in% "label"]
y  <- as.factor(df$label)

cat(sprintf("Dataset: %d obs × %d features\n", nrow(X), ncol(X)))
cat(sprintf("Class balance: %.2f%% positive\n\n", mean(df$label) * 100))

# ── Run Boruta ─────────────────────────────────────────────────────────────────
set.seed(42)
t0 <- proc.time()

boruta_result <- Boruta(
  x         = X,
  y         = y,
  maxRuns   = 100,
  pValue    = 0.01,
  mcAdj     = TRUE,          # Bonferroni correction
  doTrace   = 0
)

elapsed <- (proc.time() - t0)[["elapsed"]]

# ── Results ───────────────────────────────────────────────────────────────────
dec <- boruta_result$finalDecision
confirmed <- names(dec[dec == "Confirmed"])
rejected  <- names(dec[dec == "Rejected"])
tentative <- names(dec[dec == "Tentative"])

cat(sprintf("=== Boruta (R) results (%d iterations, %.2fs) ===\n",
            boruta_result$timeTaken |> is.null() |> ifelse(NA, length(boruta_result$ImpHistory)),
            elapsed))
cat(sprintf("  Confirmed  (%2d): %s\n", length(confirmed), paste(confirmed, collapse=", ")))
cat(sprintf("  Rejected   (%2d): %s\n", length(rejected),  paste(rejected,  collapse=", ")))
cat(sprintf("  Tentative  (%2d): %s\n", length(tentative), paste(tentative, collapse=", ")))
cat(sprintf("\n  Elapsed: %.3fs\n", elapsed))
