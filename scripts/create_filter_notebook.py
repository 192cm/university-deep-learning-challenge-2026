"""Create and execute the reproducible dataset-filtering QA notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "filtering" / "filtering_analysis.ipynb"


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Deep Challenge Math Train 필터링 검증

## tl;dr

- 원본 17,000개 샘플을 행 단위로 판정하고 고위험 후보를 원문 검토했다.
- 명백한 결함이 있는 472개를 제거하고 16,528개를 보존했다.
- 보존 우선 원칙을 사용했으므로, 남아 있는 모든 수학 라벨의 정답성을 증명하는 전수 해설 검증은 아니다.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

목적은 텍스트만으로 풀 수 없는 문항, 불필요하거나 소실된 문구가 섞인 문항, 단일 정수 라벨과 구조적으로 맞지 않는 문항, 그리고 독립 계산으로 명백히 확인된 라벨 불일치를 제외하는 것이다.

판정은 필수 필드·ID·정답 형식 검증, 제어문자/마크업/외부 시각자료 신호 탐지, 번역·운영 지시문 탐지, 다중 출력 구조 탐지, 수동 검토 목록, 독립 계산한 라벨 불일치 목록을 결합했다. 불확실한 행은 삭제하지 않았다.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

import pandas as pd

ROOT = Path.cwd()
RAW = ROOT / 'data' / 'deep_chal_math_train.csv'
FILTERED = ROOT / 'data' / 'deep_chal_math_train_filtered.csv'
AUDIT = ROOT / 'data' / 'deep_chal_math_train_filter_audit.csv'
SUMMARY = ROOT / 'report' / 'filtering' / 'filter_summary.json'

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

raw = read_csv(RAW)
filtered = read_csv(FILTERED)
audit = read_csv(AUDIT)
summary = json.loads(SUMMARY.read_text(encoding='utf-8'))

print('source_sha256:', hashlib.sha256(RAW.read_bytes()).hexdigest())
print('rows:', {'raw': len(raw), 'filtered': len(filtered), 'audit': len(audit)})
"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """profile = {
    'columns': list(raw[0]),
    'unique_ids': len({r['id'] for r in raw}),
    'blank_id': sum(not r['id'].strip() for r in raw),
    'blank_question': sum(not r['question'].strip() for r in raw),
    'blank_answer': sum(not r['answer'].strip() for r in raw),
    'exact_duplicate_questions': len(raw) - len({r['question'] for r in raw}),
    'integer_answers': sum(r['answer'].lstrip('-').isdigit() for r in raw),
}
pd.DataFrame([profile]).T.rename(columns={0: 'value'})
"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """raw_ids = [r['id'] for r in raw]
filtered_ids = [r['id'] for r in filtered]
audit_by_id = {r['id']: r for r in audit}
removed_ids = {r['id'] for r in audit if r['decision'] == 'remove'}

assert len(raw) == 17_000
assert len(audit) == len(raw)
assert len(filtered) + len(removed_ids) == len(raw)
assert len(raw_ids) == len(set(raw_ids))
assert not (set(filtered_ids) & removed_ids)
assert filtered_ids == [row_id for row_id in raw_ids if row_id not in removed_ids]
assert hashlib.sha256(RAW.read_bytes()).hexdigest() == summary['source']['sha256']
assert len(filtered) == summary['decision_counts']['keep']
assert len(removed_ids) == summary['decision_counts']['remove']

reason_counts = Counter(
    r['primary_reason'] for r in audit if r['decision'] == 'remove'
)
reason_table = pd.DataFrame(
    [
        {
            'reason': reason,
            'removed_rows': count,
            'share_of_removed_pct': round(100 * count / len(removed_ids), 2),
            'share_of_source_pct': round(100 * count / len(raw), 3),
        }
        for reason, count in reason_counts.most_common()
    ]
)
print('All integrity assertions passed.')
reason_table
"""
        ),
        nbf.v4.new_code_cell(
            """label_mismatches = pd.DataFrame(
    [
        {
            'id': r['id'],
            'source_label': r['answer'],
            'evidence': r['evidence'],
        }
        for r in audit
        if 'verified_label_mismatch' in r['reason_codes']
    ]
)
label_mismatches
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

가장 큰 제거 사유는 외부 시각자료 의존과 단일 정수 라벨에 맞지 않는 다중 출력 문항이다. 행별 감사 로그에는 보존/제거 결정, 모든 사유 코드, 근거, 신뢰도, 질문 해시를 남겨 후속 검토와 되돌리기가 가능하다.

### Limitations

- 모든 행은 판정 파이프라인을 통과했고 고위험 후보는 원문 검토했지만, 16,528개 보존 문항 각각에 대해 완전한 수학 풀이를 새로 작성한 것은 아니다.
- 라벨 불일치 제거는 명백한 사례만 포함하는 고정밀 목록이며, 보존 라벨 전체의 무오류를 보장하지 않는다.
- 불확실한 문항은 과삭제를 피하기 위해 보존했다.
"""
        ),
    ]
    return nb


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
