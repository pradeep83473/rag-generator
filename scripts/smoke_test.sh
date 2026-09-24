#!/usr/bin/env bash
# End-to-end check against a running server with a real OpenAI key.
# Creates one knowledge base per folder in samples/ and asks each a question,
# plus an out-of-scope question that should be declined.
# Usage: ./scripts/smoke_test.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8000}"

create_kb() {  # $1 = sample folder -> prints collection id
  local id
  id=$(curl -sf -XPOST "$BASE/api/collections" -H 'content-type: application/json' \
       -d "{\"name\":\"Sample: $1\"}" | python -c "import sys,json;print(json.load(sys.stdin)['id'])")
  local args=(); for f in samples/"$1"/*; do args+=(-F "files=@$f"); done
  curl -sf "${args[@]}" "$BASE/api/collections/$id/documents" > /dev/null
  echo "$id"
}

ask() {  # $1 = collection id, $2 = question
  echo "Q: $2"
  curl -sf -XPOST "$BASE/api/collections/$1/query" -H 'content-type: application/json' \
       -d "{\"question\":\"$2\"}" | python -c "import sys,json;d=json.load(sys.stdin);print('A:',d['answer']);print('   grounded:',d['grounded']);print()"
}

HANDBOOK=$(create_kb handbook)
PRODUCT=$(create_kb product)
ask "$HANDBOOK" "How many days of annual leave can I carry over, and until when?"
ask "$PRODUCT"  "What does error E-4471 mean?"
ask "$PRODUCT"  "How long is the warranty and can it be extended?"
ask "$PRODUCT"  "What is the capital of France?"
