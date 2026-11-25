#!/usr/bin/env bash
# Kiểm tra toàn bộ dự án, ghi AUDIT_REPORT.txt, tạo bản gọn và nén để gửi.
# Chạy: bash scripts/audit_pack.sh   (từ bất kỳ thư mục nào)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
NAME="$(basename "$ROOT")"
OUT_BASE="${OUT_BASE:-$HOME/Downloads}"
SLIM="$OUT_BASE/$NAME slim"
ZIP="$OUT_BASE/${NAME// /_}_slim.zip"
PY="python3"; [ -x ".venv/bin/python" ] && PY=".venv/bin/python"
REPORT="$ROOT/AUDIT_REPORT.txt"

echo "Đang kiểm tra (khoảng 1 phút: có huấn luyện lại mô hình)..."
{
echo "=== AUDIT $(date '+%Y-%m-%d %H:%M') ==="
echo "Python: $($PY --version 2>&1)"

echo; echo "--- 1. File thiếu / rỗng ---"
for f in app.py requirements.txt README.md tests/test_e2e.py \
         src/predict.py src/evaluate.py src/export.py \
         models/ridge.joblib models/rf.joblib models/mlp.pt models/preprocessor.joblib \
         outputs/metrics.csv outputs/evaluation_report.md; do
  if [ ! -e "$f" ]; then echo "THIẾU  $f"
  elif [ "$(wc -c < "$f" | tr -d ' ')" -lt 30 ]; then echo "RỖNG   $f"
  else echo "OK     $f"; fi
done

echo; echo "--- 2. Gate từng giai đoạn (chỉ hiện dòng FAIL và kết luận) ---"
S=0
for n in 1 2 3 4 5; do
  out=$($PY scripts/gate.py "$n" 2>&1)
  echo "$out" | grep -E "^(FAIL|GATE)"
  r=$(echo "$out" | grep -E "^GATE" | sed -E 's/.*\(([0-9]+)\/([0-9]+)\).*/\1 \2/')
  [ -z "$r" ] && r="0 1"
  S=$(echo "$r" | awk -v s="$S" '{print s + ($2>0 ? $1/$2 : 0)}')
done

echo; echo "--- 3. Pytest toàn bộ ---"
$PY -m pytest -q tests 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed|no tests"

echo; echo "--- 4. Tiến độ ước lượng (trung bình tỉ lệ PASS của 5 gate) ---"
echo "$S" | awk '{printf "%.0f%%\n", $1/5*100}'

echo; echo "--- 5. Thiếu sót đã biết (cố định) ---"
echo "- Chưa có train.csv/test.csv thật và Technical Handover Document: mọi số liệu là SYNTHETIC."
echo "- schema.py là giả định; khi có dữ liệu thật phải sửa schema.py rồi train/evaluate lại."
echo "- models.py: torch.load(weights_only=False) chỉ nên nạp file mlp.pt do chính dự án tạo."
echo "- Test tập trung ở pipeline; chưa có kiểm thử trên dữ liệu thật hay kiểm tra hiệu năng."
} > "$REPORT" 2>&1

rm -rf "$SLIM" "$ZIP"; mkdir -p "$SLIM"
tar -C "$ROOT" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.git' \
  --exclude='.DS_Store' --exclude='rf.joblib' --exclude='*BUNDLE*' \
  -cf - . | tar -C "$SLIM" -xf -

ALL=$(du -sk "$ROOT" | cut -f1)
VENV=$(du -sk "$ROOT/.venv" 2>/dev/null | cut -f1)
BEFORE=$((ALL - ${VENV:-0}))
AFTER=$(du -sk "$SLIM" | cut -f1)
{
echo; echo "--- 6. Dung lượng (không tính .venv) ---"
echo "Gốc: ${BEFORE} KB | Bản gọn: ${AFTER} KB"
echo "Đã bỏ: .venv, cache, models/rf.joblib (tạo lại bằng: python -m src.train), prompts/*BUNDLE*"
} >> "$REPORT"
cp "$REPORT" "$SLIM/AUDIT_REPORT.txt"
( cd "$OUT_BASE" && zip -qr "$ZIP" "$NAME slim" -x "*.DS_Store" )

echo; cat "$REPORT"
echo; echo "XONG. Zip $(du -k "$ZIP" | cut -f1) KB. Gửi file: $ZIP"
