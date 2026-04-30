# simulation_mpcp

MPCP(Multi-Processor Ceiling Protocol) 기반 Segmented Inference 태스크셋에 대한 스케줄러빌리티 분석 및 시뮬레이션 도구.

---

## 디렉토리 구조

```
simulation_mpcp/
├── task.py                          # 태스크 데이터 모델
├── analysis.py                      # RTA 알고리즘 구현
├── utils.py                         # 공통 유틸리티
│
├── generate_task_set.py             # 태스크셋 생성기
├── generate_task_set.yaml           # 생성 파라미터 설정
│
├── simulation.py                    # 시뮬레이션 메인 러너
├── simulation.yaml                  # 시뮬레이션 설정
│
├── trace.py                         # 분할 추적 시각화
├── summarize_trace_lr.py            # LR 슬로프 요약
├── plot_unsched_block_count.py      # 블록 수 분포 플롯
├── postprocess_top2_from_logs.py    # 로그 후처리
├── find_tolerance_mismatch_indices.py  # 불일치 인덱스 탐색
│
├── run_overnight.sh                 # 다중 설정 일괄 실행
├── stop_simulation.sh               # 실행 중 시뮬레이션 강제 종료
│
├── legacy/                          # 이전 구현 보관
│   ├── analysis_old.py
│   ├── simulation_old.py
│   └── simulation_old.yaml
│
└── overnight/                       # 일괄 실행용 설정 파일 모음
    └── ...
```

---

## 태스크 모델

### `InferenceSegment`

GPU 추론 세그먼트 하나를 표현한다. 내부적으로 `base_block_list`(고정 참조)와 `splitting_config`(이진 벡터)로 현재 분할 형태를 관리한다.

```python
from task import InferenceSegment

seg = InferenceSegment(
    G_segment=100,              # 원시 GPU 시간 (overhead 미포함)
    max_block_count=10,         # 최대 분할 가능 블록 수
    per_splitting_overhead=5,   # 블록 간 분리 overhead
)

seg.G            # 현재 총 GPU 시간 (overhead 포함)
seg.size         # 현재 블록 수
seg.overhead     # 현재 총 overhead = (size-1) * per_splitting_overhead
seg.G_block_list # 블록 크기 리스트

seg.split_segment(3)               # 3개 그룹으로 분할
seg.split_by_config([1,0,1,0,...]) # splitting_config 직접 적용
```

**splitting_config 규칙**
- 길이: `max_block_count - 1`
- `1`: 인접한 두 base block 사이를 분리
- `0`: 인접한 두 base block을 병합
- overhead는 마지막 블록을 제외한 모든 블록에 추가됨

### `SegInfTask`

여러 CPU+GPU 세그먼트로 구성된 태스크.

```python
from task import SegInfTask

task = SegInfTask(
    id=0,
    segment_list=[
        {
            'C': 100,                    # CPU 실행 시간
            'G_segment': 200,            # GPU 시간 (0이면 CPU-only 세그먼트)
            'max_block_count': 10,       # 최대 블록 수
            'per_splitting_overhead': 5, # 분리 overhead
        },
        {
            'C': 50,
            'G_segment': 0,              # 마지막 CPU-only 세그먼트
            'max_block_count': 1,
            'per_splitting_overhead': 5,
        },
    ],
    period=1000,
    deadline=1000,
    priority=1/1000,
    cpu=0,
)

task.C              # 전체 CPU 시간
task.G              # 전체 GPU 시간 (overhead 포함)
task.max_G_block    # 최대 블록 크기
task.G_segment_list # inference segment별 G_block_list 목록
task.m              # 실행 세그먼트 수 (CPU + GPU)

task.split_segment(idx=0, n=3)                  # 0번 segment를 3개로 분할
task.split_by_config(idx=0, splitting_config=[1,0,...])  # config 직접 적용
```

---

## 1단계: 태스크셋 생성

### 설정 파일 편집

```yaml
# generate_task_set.yaml

output_dir: task_set_list/utilization   # 출력 디렉토리
n_task_sets: 100                        # 이용률 포인트당 태스크셋 수

number_of_cpu_range: [4, 4]             # CPU 수 범위
utilization_per_cpu_range: [0.05, 0.25] # CPU당 이용률 범위
utilization_step: 0.05                  # 이용률 스텝

number_of_tasks_per_cpu_range: [1, 3]   # CPU당 태스크 수 범위
period_range: [1000, 10000]             # 주기 범위
G_ratio_range: [0.1, 0.8]        # G 비율 범위 (C+G 중 G의 비율)
number_of_inference_segments_range: [0, 3]  # inference segment 수 범위
max_block_count_range: [10, 100]        # 최대 블록 수 범위

per_splitting_overhead: 5               # 분리 overhead 값
```

### 실행

```bash
python3 generate_task_set.py
```

**출력** (`output_dir` 디렉토리 내):
```
task_set_list/utilization/
├── generate_task_set.yaml         # 재현용 설정 스냅샷
├── task_set_list_u0.05.pkl        # 이용률 0.05 태스크셋 (pickle)
├── task_set_list_u0.05.json       # 이용률 0.05 태스크셋 (JSON, 확인용)
├── simulation_log_u0.05.txt       # 생성 로그
├── task_set_list_u0.1.pkl
...
```

---

## 2단계: 시뮬레이션

### 설정 파일 편집

```yaml
# simulation.yaml

task_set_list_dir_path: task_set_list/utilization  # generate_task_set 출력 경로

# 결과 디렉토리 prefix (선택)
# result_dir_prefix: my_experiment

# RTA 메서드 개별 on/off
enable_RTA_SS_single: true
enable_RTA_SS_max: true
enable_RTA_SS_tol: true
enable_RTA_SS_tol_fb: false
enable_RTA_SS_tol_fb_early: true
enable_RTA_UNI_tol_fb: true
enable_RTA_UNI_opt: false
enable_RTA_UNI_heu: true
enable_RTA_SS_opt: false
enable_RTA_SS_heu: true
```

### 실행

```bash
python3 simulation.py
```

**결과 디렉토리** (`result/<YYMMDD-HHMM>[_prefix]/`):
```
result/260429-1205_my_experiment/
├── simulation.yaml                     # 설정 스냅샷
├── generate_task_set.yaml
├── simulation_runtime.txt              # 완료/오류 상태
├── schedulability_ratio_by_utilization.png  # 메인 플롯
├── rta_logs/
│   ├── rta_task_set_list_u0.05.log    # 이용률별 RTA 결과 로그
│   └── ...
└── task_sets/
    └── ...                             # 사용한 태스크셋 복사본
```

### 중단 및 재개

```bash
# 실행 중 시뮬레이션 종료
bash stop_simulation.sh

python3 simulation.py
```

---

## 3단계: 트레이스 시각화

시뮬레이션 결과에서 분할 과정을 추적하여 CSV/PNG를 생성한다.

### tol_max 모드 — TOL_MAX 분할 추적

```bash
python3 trace.py \
  --mode tol_max \
  --run-dir result/260429-1205_my_experiment \
  --utilization 0.2 \
  --output-root trace/
```

### r_best 모드 — Optimistic vs Actual 응답시간 비교

```bash
python3 trace.py \
  --mode r_best \
  --run-dir result/260429-1205_my_experiment \
  --utilization 0.2 \
  --tol-method RTA_SS_tol_fb_rbest \
  --output-root trace/
```

### lr 모드 — LR 기울기 플롯

```bash
python3 trace.py \
  --mode lr \
  --run-dir result/260429-1205_my_experiment \
  --utilization 0.2 \
  --n-list 2,5,10 \
  --subset unsched \
  --output-root trace/
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | `tol_max` / `r_best` / `lr` | `tol_max` |
| `--run-dir` | 시뮬레이션 결과 디렉토리 | (필수) |
| `--utilization` | 대상 이용률 | `0.2` |
| `--tol-method` | 분석할 RTA 메서드명 | 자동 선택 |
| `--task-set-idx` | 특정 태스크셋 인덱스만 처리 | 전체 |
| `--max-workers` | 병렬 처리 worker 수 | 자동 |
| `--subset` | `unsched` / `sched` / `all` | `unsched` |
| `--n-list` | LR 윈도우 크기 목록 (lr 모드) | `10` |

**출력** (`output-root/<trace_type>_<timestamp>_u<util>/`):
```
schedulable/
│   trace_rbest_u0.2_idx42_sched_trace.csv
│   trace_rbest_u0.2_idx42_sched_trace.png
unschedulable/
│   trace_rbest_u0.2_idx7_unsched_trace.csv
│   trace_rbest_u0.2_idx7_unsched_trace.png
```

---

## 일괄 실행 (overnight)

여러 실험 설정을 순차적으로 자동 실행한다.

### overnight 설정 파일 구조

`overnight/` 디렉토리의 YAML 파일은 `generate_task_set.yaml`과 동일한 포맷이다. 파일 이름 앞의 번호 순으로 실행된다.

```
overnight/
├── 1_base.yaml              # 기본 설정
├── 2_tasks_per_cpu_1.yaml   # CPU당 태스크 수 = 1
├── 3_tasks_per_cpu_3.yaml   # CPU당 태스크 수 = 3
├── 4_G_ratio_0.1.yaml       # G 비율 = 0.1
├── 5_G_ratio_0.8.yaml       # G 비율 = 0.8
├── 6_inference_segments_1.yaml  # inference segment 수 = 1
├── 7_inference_segments_3.yaml  # inference segment 수 = 3
├── 8_max_block_cnt_10.yaml  # max_block_count = 10
├── 9_max_block_cnt_100.yaml # max_block_count = 100
├── 10_overhead_0.yaml       # per_splitting_overhead = 0
├── 11_overhead_1.yaml       # per_splitting_overhead = 1
└── 12_overhead_10.yaml      # per_splitting_overhead = 10
```

### 실행

```bash
bash run_overnight.sh
```

각 케이스마다 자동으로 다음을 수행한다:
1. `generate_task_set.py` 실행 (태스크셋 생성)
2. `simulation.py` 실행 (1차 시도)
3. 미완료 시 `simulation.py` 재개 실행 (최대 1회)
4. 완료된 결과에 대해 `trace.py --mode r_best` 실행

**로그 위치** (`overnight_runs/<timestamp>_from_configs/`):
```
overnight_runs/260429-120500_from_configs/
├── 1_base_generate.log
├── 1_base_simulation.log
├── 1_base_simulation_resume1.log
├── 1_base_trace_rbest.log
├── 1_base_result_dir.txt       # 결과 디렉토리 경로 기록
├── 2_tasks_per_cpu_1_generate.log
...
```

### 강제 종료

```bash
bash stop_simulation.sh
```

실행 중인 `simulation.py` 프로세스를 찾아 종료한다. SIGTERM 후 5초 내 미종료 시 SIGKILL을 보낸다.

---

## 후처리 도구

### `plot_unsched_block_count.py` — 비스케줄 태스크셋의 profiling count 분포

시뮬레이션 결과에서 스케줄 불가능 태스크셋들의 profiling count 분포를 boxplot으로 생성한다.

```bash
python3 plot_unsched_block_count.py \
  --run-dir result/260429-1205_my_experiment
```

### `postprocess_top2_from_logs.py` — TOL vs LR 메서드 성능 비교

여러 이용률 포인트의 로그를 분석하여 TOL_MAX와 LR-N 메서드 간 profiling count 비교 CSV를 생성한다.

```bash
python3 postprocess_top2_from_logs.py \
  --run-dir result/260429-1205_my_experiment \
  --output comparison.csv
```

### `summarize_trace_lr.py` — LR 트레이스 요약

`trace.py`가 생성한 CSV 파일들에서 LR 기울기 통계를 집계한다.

```bash
python3 summarize_trace_lr.py \
  --trace-root trace/trace_r_best_<timestamp>_u0.2
```

### `find_tolerance_mismatch_indices.py` — TOL/MAX 불일치 인덱스 탐색

`max_splitting`은 스케줄 가능하지만 `tolerance_based_splitting`은 스케줄 불가능한 태스크셋 인덱스를 RTA 로그에서 찾아낸다.

```bash
python3 find_tolerance_mismatch_indices.py \
  --rta-log result/260429-1205_my_experiment/rta_logs/rta_task_set_list_u0.2.log
```

---

## RTA 메서드 목록

| 메서드 | 설명 |
|--------|------|
| `RTA_SS_single` | 기본 MPCP RTA (분할 없음) |
| `RTA_SS_max` | 모든 segment를 최대로 분할 |
| `RTA_SS_tol` | tolerance 기반 분할 |
| `RTA_SS_tol_fb` | TOL 실패 시 MAX로 fallback |
| `RTA_SS_tol_fb_early` | TOL_MAX + R 증가 시 조기 종료 |
| `RTA_UNI_tol_fb` | UNI 분석 기반 TOL fallback |
| `RTA_UNI_opt` | UNI 변환 후 optimal splitting config 탐색 |
| `RTA_UNI_heu` | UNI 변환 후 heuristic splitting config 탐색 |
| `RTA_SS_opt` | SS optimal splitting config 탐색 |
| `RTA_SS_heu` | SS heuristic splitting config 탐색 |

---

## 일반적인 실험 플로우

```bash
# 1. 태스크셋 생성
python3 generate_task_set.py

# 2. 시뮬레이션
python3 simulation.py

# 3. r_best 트레이스 생성 (원하는 이용률 포인트 지정)
python3 trace.py \
  --mode r_best \
  --run-dir result/<result_dir> \
  --utilization 0.2

# 4. 불일치 인덱스 확인 (선택)
python3 find_tolerance_mismatch_indices.py \
  --rta-log result/<result_dir>/rta_logs/rta_task_set_list_u0.2.log
```

다중 실험은 overnight 설정 파일을 추가하고 `run_overnight.sh`로 일괄 실행한다.


## Tolerance tracing하기
```
python3 plot_min_tolerance_trace.py \
  --run-dir result/260429-1205_my_experiment \
  --utilization 0.7 \
  --combined
```
