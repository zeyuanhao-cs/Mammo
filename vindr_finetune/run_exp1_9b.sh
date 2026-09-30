#!/usr/bin/env bash
# Run inside the Mammo Slurm image with the mammo bind mount.
set -euo pipefail
: "${JOB_ID:?Slurm-Web JOB_ID is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
export USER=yu.w LOGNAME=yu.w PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/me/mammo/cache/huggingface
export XDG_CACHE_HOME=/data/me/mammo/cache/xdg
export TMPDIR=/data/me/mammo/cache/tmp
export RUN_DIR="/data/me/mammo/runs/exp1-9b-${JOB_ID}"
export ADAPTER_DIR="/data/me/mammo/qwen3.5-9b-balanced-2to1-job-${JOB_ID}"
export PRED_DIR="/data/me/mammo/outputs/direct_qwen3.5-9b-balanced-2to1-job-${JOB_ID}_pixels786432_sdpa"
if test -e "$RUN_DIR" || test -e "$ADAPTER_DIR" || test -e "$PRED_DIR"; then
    echo 'phase=refused reason=output_already_exists'
    exit 73
fi
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$RUN_DIR"
trap 'rc=$?; echo "phase=failed exit_code=$rc"; exit "$rc"' ERR

python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import yaml

source = Path(os.environ['SCRIPT_DIR'])
run = Path(os.environ['RUN_DIR'])
manifest = {'job_id': os.environ['JOB_ID'], 'git_commit': os.environ.get('MAMMO_COMMIT'), 'datasets': {}}
sets = {}
for name, count, expected_hash in [
    ('train_balanced_2to1', 4657, '93d53adefac511f8d355e5dc72ead78526a54e1c2e1d55a6aba0890bc1584ce9'),
    ('direct_test', 4000, '6eddbabedff8e37df2f5629585b4947fea3cbea6df9f61d9a1fed13548b8249e'),
]:
    path = source / 'data' / f'{name}.json'
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == expected_hash, f'{name}: unexpected data version'
    rows = json.loads(blob)
    assert len(rows) == count, f'{name}: unexpected count'
    images = set()
    for row in rows:
        assert len(row['images']) == row['instruction'].count('<image>') == 1
        assert isinstance(json.loads(row['output']), dict)
        image = (Path('/mammo') / row['images'][0]).resolve()
        assert image.is_relative_to(Path('/mammo/images_png')), 'image path outside dataset'
        assert image.is_file() and os.access(image, os.R_OK), 'image absent or unreadable'
        images.add(str(image))
    sets[name] = images
    manifest['datasets'][name] = {'rows': len(rows), 'unique_images': len(images), 'sha256': digest}
assert not sets['train_balanced_2to1'] & sets['direct_test'], 'train/test image overlap'
config = yaml.safe_load((source / 'trial_9b_balanced.yaml').read_text())
config['dataset_dir'] = str(source)
config['output_dir'] = os.environ['ADAPTER_DIR']
model = Path(config['model_name_or_path'])
assert os.access(model / 'config.json', os.R_OK), 'model config unreadable'
index = json.loads((model / 'model.safetensors.index.json').read_text())
for shard in set(index['weight_map'].values()):
    path = model / shard
    assert path.is_file() and path.stat().st_size and os.access(path, os.R_OK), 'model shard unreadable'
(run / 'train.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
(run / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('phase=preflight_complete train_rows=4657 test_rows=4000 missing_images=0 overlap=0', flush=True)
PY

echo "phase=train_start job_id=${JOB_ID} gpu_physical=${MAMMO_GPU_PHYSICAL:?}"
llamafactory-cli train "$RUN_DIR/train.yaml" > "$RUN_DIR/train.log" 2>&1
test -s "$ADAPTER_DIR/adapter_config.json"
test -s "$ADAPTER_DIR/adapter_model.safetensors"
echo "phase=train_complete job_id=${JOB_ID}"
echo "phase=infer_start job_id=${JOB_ID}"
python3 "$SCRIPT_DIR/infer_lora.py" \
    --prompt direct --split test \
    --model /data/models/Qwen3.5-9B --adapter "$ADAPTER_DIR" \
    --image-root /mammo --image-max-pixels 786432 --image-min-pixels 262144 \
    --max-new-tokens 256 --flash-attn sdpa > "$RUN_DIR/infer.log" 2>&1

python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ['PRED_DIR']) / 'predictions.jsonl'
count = failures = invalid = 0
indices = set()
with path.open() as stream:
    for line in stream:
        row = json.loads(line)
        count += 1
        indices.add(row['index'])
        failures += bool(row.get('error'))
        invalid += not isinstance(row.get('eval_json'), dict)
summary = {'rows': count, 'unique_indices': len(indices), 'runtime_errors': failures,
           'invalid_json': invalid, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(Path(os.environ['RUN_DIR']) / 'inference_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
assert count == 4000 and indices == set(range(4000)), 'incomplete inference'
assert failures == 0, 'inference has runtime errors; inspect aggregate error categories'
PY
echo "phase=complete job_id=${JOB_ID}"
