#!/usr/bin/env sh
# Full pipeline: train both models, build the search index, run every evaluation.
# Uses EMBEDDING_BACKEND from the environment (sbert or tfidf).
set -eu
cd "$(dirname "$0")/.."
ART=${ARTIFACT_ROOT:-artifacts}

echo "== 1/5 train base model"
python -m comms_triage.classify.train --out "$ART/base"
echo "== 2/5 train robust model (with informal and typo augmentation)"
python -m comms_triage.classify.train --out "$ART/robust" --augment-robust ${FEEDBACK:+--feedback "$FEEDBACK"}
echo "== 3/5 build knowledge index"
MODEL_DIR="$ART/robust" python -m comms_triage.retrieve.index
echo "== 4/5 robustness"
python eval/robustness.py --models "$ART/base" "$ART/robust"
echo "== 5/5 retrieval and extraction"
MODEL_DIR="$ART/robust" python eval/retrieval_eval.py
MODEL_DIR="$ART/robust" python eval/extraction_eval.py
echo "done: serving model is $ART/robust"
