#!/usr/bin/env bash
set -euo pipefail
: "${JOB_ID:?Slurm-Web JOB_ID is required}"
: "${MAMMO_COMMIT:?Pinned code commit is required}"
: "${MAMMO_GPU_PHYSICAL:?Pinned physical GPU is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export USER=yu.w LOGNAME=yu.w
export HF_HOME=/data/me/mammo/cache/huggingface XDG_CACHE_HOME=/data/me/mammo/cache/xdg
export TMPDIR=/data/me/mammo/cache/tmp
export RUN_DIR="/data/me/mammo/runs/exp2-4b-thinking-${JOB_ID}"
export ADAPTER_DIR="/data/me/mammo/qwen3.5-4b-thinking-balanced-2to1-job-${JOB_ID}"
export PRED_DIR="/data/me/mammo/outputs/direct_qwen3.5-4b-thinking-balanced-2to1-job-${JOB_ID}_pixels786432_sdpa_thinking"
export NINE_PRED=/data/me/mammo/outputs/direct_qwen3.5-9b-balanced-2to1-job-12080_pixels786432_sdpa/predictions.jsonl
export BASELINE_PRED=/mammo/.cache/mammo-benchmarks/4b-2to1-20e980b-predictions.jsonl
if test -e "$RUN_DIR" || test -e "$ADAPTER_DIR" || test -e "$PRED_DIR"; then
    echo 'phase=refused reason=output_already_exists'
    exit 73
fi
mkdir -p "$RUN_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
trap 'rc=$?; echo "phase=failed exit_code=$rc"; exit "$rc"' ERR

python3 - <<'PY'
import hashlib,json,os
from pathlib import Path
import yaml
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer

source=Path(os.environ['SCRIPT_DIR']); run=Path(os.environ['RUN_DIR'])
config=yaml.safe_load((source/'trial_4b_thinking.yaml').read_text())
dataset=Path(config['dataset_dir']); file=dataset/'train_balanced_2to1_thinking.json'
blob=file.read_bytes(); sha=hashlib.sha256(blob).hexdigest()
summary=json.loads((dataset/'summary.json').read_text())
assert sha==summary['output_sha256']=='8152daecd7a1dd16ddcd5033fa40eb111c3510d6d91be08287951960c7068707'
assert summary['state']=='complete' and summary['output_rows']==4657 and summary['excluded_rows']==0
assert summary['verdict_counts']=={'accept':0,'revised':2,'reject':0}
rows=json.loads(blob); assert len(rows)==4657
original=json.loads((source/'data/train_balanced_2to1.json').read_text())
test_file=source/'data/direct_test.json';test_blob=test_file.read_bytes()
assert hashlib.sha256(test_blob).hexdigest()=='6eddbabedff8e37df2f5629585b4947fea3cbea6df9f61d9a1fed13548b8249e'
test=json.loads(test_blob); assert len(test)==4000
train_images=set();test_images=set(); pairs={}
for row, old in zip(rows,original):
    assert row['images']==old['images'] and row['instruction'].count('<image>')==len(row['images'])==1
    assert row['output'].startswith('<think>') and row['output'].endswith('</think>\n'+old['output'])
    assert row['output'].count('<think>')==row['output'].count('</think>')==1
    train_images.update(row['images']);pairs.setdefault((row['images'][0],row['output']),row)
for row in test:test_images.update(row['images'])
assert len(train_images)==2948 and not train_images & test_images
for image in train_images | test_images:
    p=(Path('/mammo')/image).resolve()
    assert p.is_relative_to(Path('/mammo/images_png')) and p.is_file() and os.access(p,os.R_OK)
model=Path(config['model_name_or_path']); index=json.loads((model/'model.safetensors.index.json').read_text())
assert all((model/s).is_file() and (model/s).stat().st_size for s in set(index['weight_map'].values()))
tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
template=get_template_and_fix_tokenizer(tokenizer,DataArguments(template='qwen3_5',enable_thinking=True))
max_text=0
for row in pairs.values():
    prompt,answer=template.encode_oneturn(tokenizer,[{'role':'user','content':row['instruction']},
                                                  {'role':'assistant','content':row['output']}])
    decoded=tokenizer.decode(answer)
    assert '</think>' in decoded and decoded.rsplit('</think>',1)[1].strip().startswith('{')
    max_text=max(max_text,len(prompt)+len(answer))
# Conservative vision-token allowance exceeds the 786432-pixel setting's grid.
assert max_text+2048<=config['cutoff_len'], 'thinking sequence may be truncated'
assert hashlib.sha256(Path(os.environ['NINE_PRED']).read_bytes()).hexdigest()=='3f327ddab60e8e0427c393edec29e9c20ddd8fc1183f9300bda2243a84dcc417'
baseline=Path(os.environ['BASELINE_PRED']).read_bytes()
assert hashlib.sha1(b'blob '+str(len(baseline)).encode()+b'\0'+baseline).hexdigest()=='c0ebb1017f883159866b0ee222242036958439e5'
old=[json.loads(line) for line in baseline.splitlines() if line.strip()]
assert len(old)==500 and {r['index'] for r in old}==set(range(500))
assert all(json.loads(r['ground_truth'])==json.loads(test[r['index']]['output']) for r in old)
config['output_dir']=os.environ['ADAPTER_DIR']
(run/'train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(run/'manifest.json').write_text(json.dumps({'job_id':os.environ['JOB_ID'],'git_commit':os.environ['MAMMO_COMMIT'],
    'gpu_physical':os.environ['MAMMO_GPU_PHYSICAL'],'train_rows':4657,'train_unique_images':2948,
    'train_sha256':sha,'test_rows':4000,'test_sha256':hashlib.sha256(test_blob).hexdigest(),
    'train_test_overlap':0,'max_text_tokens':max_text,'vision_token_allowance':2048,
    'cutoff_len':config['cutoff_len'],'epochs':2,'enable_thinking':True,'infer_max_new_tokens':4096},indent=2))
print(json.dumps({'phase':'preflight_complete','train_rows':4657,'test_rows':4000,'overlap':0,
                  'max_text_tokens':max_text,'cutoff_len':config['cutoff_len'],'enable_thinking':True}),flush=True)
PY

echo "phase=train_start job_id=${JOB_ID} gpu_physical=${MAMMO_GPU_PHYSICAL}"
llamafactory-cli train "$RUN_DIR/train.yaml" > "$RUN_DIR/train.log" 2>&1
test -s "$ADAPTER_DIR/adapter_config.json"
test -s "$ADAPTER_DIR/adapter_model.safetensors"
echo "phase=train_complete job_id=${JOB_ID}"

infer() {
    python3 "$SCRIPT_DIR/infer_lora.py" --prompt direct --split test \
        --model /data/models/Qwen3.5-4B --adapter "$ADAPTER_DIR" \
        --image-root /mammo --image-max-pixels 786432 --image-min-pixels 262144 \
        --enable-thinking --max-new-tokens 4096 --flash-attn sdpa --limit "$1"
}
compare() {
    python3 "$SCRIPT_DIR/compare_thinking.py" --predictions "$PRED_DIR/predictions.jsonl" \
        --predictions-9b "$NINE_PRED" --baseline-predictions "$BASELINE_PRED" \
        --test-data "$SCRIPT_DIR/data/direct_test.json" --output-dir "$RUN_DIR/$1" --job-id "$JOB_ID"
}
echo "phase=infer_paired500_start job_id=${JOB_ID}"
infer 500 > "$RUN_DIR/infer_paired500.log" 2>&1
compare comparison_500
echo "phase=infer_full4000_start job_id=${JOB_ID}"
infer 0 > "$RUN_DIR/infer_full4000.log" 2>&1
compare comparison_full4000
echo "phase=complete job_id=${JOB_ID}"
