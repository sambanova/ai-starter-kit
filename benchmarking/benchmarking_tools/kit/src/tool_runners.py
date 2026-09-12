"""Uniform interface for running a benchmark with the Kit, vLLM, or aiperf from the Streamlit app.

Kit runs in-process (the existing SyntheticPerformanceEvaluator/CustomPerformanceEvaluator). vLLM
and aiperf are driven as subprocesses -- KEEP EACH RUNNER'S FLAGS IN SYNC WITH the corresponding
../../vllm/quickstart.sh / ../../aiperf/quickstart.sh, which remain the source of truth for how
each CLI is actually invoked. After the subprocess completes, each runner calls the exact same
convert_vllm_output.convert()/convert_aiperf_output.convert() functions those quickstart scripts
use, so a Streamlit-triggered run and a manually-run quickstart.sh produce byte-identical
_summary.json/_individual_responses.json shapes.

No Streamlit dependency -- safe to import from any UI context.

To add a future tool: implement ToolRunner, then add one entry to TOOL_RUNNERS. No other code
(UI, comparison/display logic) needs to change.
"""

import glob
import io
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, cast

from benchmarking.benchmarking_utils import get_tokenizer_model_name
from benchmarking.benchmarking_tools.kit.src.convert_aiperf_output import convert as convert_aiperf
from benchmarking.benchmarking_tools.kit.src.convert_vllm_output import convert as convert_vllm
from benchmarking.benchmarking_tools.kit.src.generate_dataset import generate_dataset
from benchmarking.benchmarking_tools.kit.src.performance_evaluation import (
    CustomPerformanceEvaluator,
    RealWorkLoadPerformanceEvaluator,
    SyntheticPerformanceEvaluator,
)

ProgressCallback = Callable[[int, int, str], None]
# Called once per line of raw subprocess output (vLLM/aiperf only -- Kit runs in-process and
# already exposes real granular progress via ProgressCallback, so it never calls this).
LogCallback = Callable[[str], None]


@dataclass
class ToolRunResult:
    tool: str
    summary_file_path: str
    individual_responses_file_path: str
    raw_output_paths: List[str] = field(default_factory=list)


class ToolRunner(Protocol):
    name: str
    display_name: str
    supports_server_metrics: bool
    supports_batching_info: bool

    def run_synthetic(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        qps: Optional[float],
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult: ...

    def run_custom(
        self,
        *,
        model_name: str,
        uploaded_dataset_path: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        num_output_tokens: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult: ...

    def run_real_workload(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        qps: float,
        qps_distribution: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult: ...

    def stop(self) -> None: ...


def _count_jsonl_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _iter_subprocess_output(proc: 'subprocess.Popen[bytes]') -> Iterator[str]:
    """Yields a subprocess's raw stdout as it's produced, one displayed "line" at a time.

    Two things a naive `for line in proc.stdout:` gets wrong for CLIs like vLLM/aiperf that render
    a tqdm-style progress bar, both confirmed empirically (not just from reading tqdm's source):

    1. tqdm redraws its bar in place using '\\r' (carriage return), not '\\n' -- and never emits a
       real '\\n' until it closes. Splitting only on '\\n' (what plain binary-mode line iteration
       does) means the entire progress bar's worth of updates gets treated as *one* line, which
       only becomes visible once some later, unrelated '\\n'-terminated output finally appears.
    2. `io.BufferedReader.read(n)` blocks until `n` bytes are available or EOF -- so even after
       fixing the splitting, reading in fixed-size chunks via `.read(n)` still silently waits for
       the buffer to fill, which for a modest-output CLI can mean it never returns anything until
       the process exits. `.read1(n)` returns after a single underlying read call (as soon as any
       data is available at all), which is what an actually-live log needs.
    """
    assert proc.stdout is not None
    # subprocess.Popen's stdout is typed generically as IO[bytes], but with stdout=PIPE it's
    # always a real io.BufferedReader at runtime, which is what actually has .read1().
    stdout = cast(io.BufferedReader, proc.stdout)
    buffer = b''
    while True:
        chunk = stdout.read1(4096)
        if not chunk:
            break
        buffer += chunk
        while True:
            idx_r = buffer.find(b'\r')
            idx_n = buffer.find(b'\n')
            if idx_r == -1 and idx_n == -1:
                break
            if idx_n != -1 and (idx_r == -1 or idx_n < idx_r):
                line, buffer = buffer[:idx_n], buffer[idx_n + 1 :]
            else:
                # Treat a '\r\n' pair as one separator, same as universal-newlines text mode.
                cut = idx_r + 2 if buffer[idx_r + 1 : idx_r + 2] == b'\n' else idx_r + 1
                line, buffer = buffer[:idx_r], buffer[cut:]
            yield line.decode(errors='replace')
    if buffer:
        yield buffer.decode(errors='replace')


def _vllm_burstiness_for_distribution(qps_distribution: str) -> str:
    """Maps Kit's qps_distribution ('constant'/'exponential') onto vLLM's --burstiness.

    vLLM has no discrete arrival-pattern choice, only a continuous burstiness knob (the shape
    parameter of a Gamma distribution sampling inter-arrival times): 1.0 is an exact Poisson/
    exponential match (a Gamma distribution with shape 1 IS an Exponential distribution --
    confirmed: matches Kit's 'exponential', which is itself a Poisson process via
    random.expovariate). There's no exact 'constant' equivalent -- a Gamma distribution's support
    is (0, inf) for any finite shape, so it can never collapse to a fixed value; a large shape only
    shrinks the coefficient of variation (CV = 1/sqrt(shape)) around the mean, approximating (not
    reproducing) constant-cadence spacing. 1000 gives CV=3.16% (e.g. +/-32ms of jitter around a 1s
    gap at qps=1) -- tight enough that it's dominated by real API/network response-time variance,
    so a further-tightened value (e.g. 10000, CV=1%) would not be distinguishable from 1000 in any
    actual benchmark output; picked as the practical ceiling rather than an arbitrarily larger
    number (documented in ../../vllm/README.md's researched mapping).
    """
    if qps_distribution == 'constant':
        return '1000'
    return '1'


def _aiperf_arrival_pattern_for_distribution(qps_distribution: str) -> str:
    """Maps Kit's qps_distribution ('constant'/'exponential') onto aiperf's --arrival-pattern.

    aiperf natively supports 'constant' (fixed-rate) and 'poisson' (confirmed live via
    `aiperf profile --help`) -- an exact match for Kit's 'constant' and 'exponential'
    (Kit's 'exponential' is itself a Poisson process) respectively.
    """
    if qps_distribution == 'constant':
        return 'constant'
    return 'poisson'


class KitToolRunner:
    """Thin reshaping of the Kit's own evaluators to the ToolRunner interface -- no new logic."""

    name = 'kit'
    display_name = 'Kit'
    supports_server_metrics = True
    supports_batching_info = True

    def __init__(self) -> None:
        self._evaluator: Optional[Any] = None

    def run_synthetic(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        qps: Optional[float],
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        # log_cb is unused here -- Kit runs in-process and already reports real granular progress
        # via progress_cb, so there's no subprocess output to stream.
        self._evaluator = SyntheticPerformanceEvaluator(
            model_name=model_name,
            results_dir=results_dir,
            multimodal_image_size='na',
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            timeout=timeout,
            llm_api='sncloud',
            api_variables={'SAMBANOVA_API_BASE': api_base, 'SAMBANOVA_API_KEY': api_key},
            user_metadata={'model_idx': 0},
        )
        self._evaluator.run_benchmark(
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            sampling_params={},
            progress_bar=progress_cb,
        )
        assert self._evaluator.summary_file_path is not None
        assert self._evaluator.individual_responses_file_path is not None
        return ToolRunResult(
            tool='kit',
            summary_file_path=self._evaluator.summary_file_path,
            individual_responses_file_path=self._evaluator.individual_responses_file_path,
        )

    def run_custom(
        self,
        *,
        model_name: str,
        uploaded_dataset_path: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        num_output_tokens: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        self._evaluator = CustomPerformanceEvaluator(
            model_name=model_name,
            results_dir=results_dir,
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            timeout=timeout,
            input_file_path=uploaded_dataset_path,
            save_response_texts=False,
            llm_api='sncloud',
            api_variables={'SAMBANOVA_API_BASE': api_base, 'SAMBANOVA_API_KEY': api_key},
        )
        self._evaluator.run_benchmark(num_output_tokens=num_output_tokens, progress_bar=progress_cb)
        assert self._evaluator.summary_file_path is not None
        assert self._evaluator.individual_responses_file_path is not None
        return ToolRunResult(
            tool='kit',
            summary_file_path=self._evaluator.summary_file_path,
            individual_responses_file_path=self._evaluator.individual_responses_file_path,
        )

    def run_real_workload(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        qps: float,
        qps_distribution: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        # num_concurrent_requests is a vLLM/aiperf-only concept here (their CLIs require an
        # explicit concurrency ceiling even in QPS-paced mode) -- Kit's own RealWorkLoadPerformanceEvaluator
        # is purely QPS-paced and doesn't take it as a real constraint, so it's ignored.
        self._evaluator = RealWorkLoadPerformanceEvaluator(
            qps=qps,
            qps_distribution=qps_distribution,
            model_name=model_name,
            results_dir=results_dir,
            multimodal_image_size='na',
            num_warmup_requests=num_warmup_requests,
            timeout=timeout,
            llm_api='sncloud',
            api_variables={'SAMBANOVA_API_BASE': api_base, 'SAMBANOVA_API_KEY': api_key},
            user_metadata={'model_idx': 0},
        )
        self._evaluator.run_benchmark(
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            sampling_params={},
            progress_bar=progress_cb,
        )
        assert self._evaluator.summary_file_path is not None
        assert self._evaluator.individual_responses_file_path is not None
        return ToolRunResult(
            tool='kit',
            summary_file_path=self._evaluator.summary_file_path,
            individual_responses_file_path=self._evaluator.individual_responses_file_path,
        )

    def stop(self) -> None:
        if self._evaluator is not None:
            self._evaluator.stop_benchmark()


class VLLMToolRunner:
    """Subprocess wrapper around `vllm bench serve` -- KEEP IN SYNC WITH ../../vllm/quickstart.sh."""

    name = 'vllm'
    display_name = 'vLLM'
    supports_server_metrics = False
    supports_batching_info = False

    def __init__(self) -> None:
        self._proc: Optional['subprocess.Popen[bytes]'] = None

    def run_synthetic(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        qps: Optional[float],
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        os.makedirs(results_dir, exist_ok=True)
        dataset_path = os.path.join(results_dir, 'vllm_synthetic_dataset.jsonl')
        generate_dataset(model_name, num_requests, num_input_tokens, dataset_path)
        return self._run(
            model_name=model_name,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            qps=qps,
            results_dir=results_dir,
            api_base=api_base,
            api_key=api_key,
            progress_cb=progress_cb,
            log_cb=log_cb,
            dataset_path=dataset_path,
            workload_mode='synthetic',
        )

    def run_custom(
        self,
        *,
        model_name: str,
        uploaded_dataset_path: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        num_output_tokens: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        # vLLM's --dataset-name custom accepts the repo's {"prompt": ...} schema directly -- no
        # conversion needed, unlike aiperf's custom-dataset-type single_turn (see AIPerfToolRunner).
        os.makedirs(results_dir, exist_ok=True)
        num_requests = _count_jsonl_lines(uploaded_dataset_path)
        return self._run(
            model_name=model_name,
            num_input_tokens=0,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            qps=None,
            results_dir=results_dir,
            api_base=api_base,
            api_key=api_key,
            progress_cb=progress_cb,
            log_cb=log_cb,
            dataset_path=uploaded_dataset_path,
            workload_mode='custom',
            no_oversample=True,
        )

    def run_real_workload(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        qps: float,
        qps_distribution: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        os.makedirs(results_dir, exist_ok=True)
        dataset_path = os.path.join(results_dir, 'vllm_synthetic_dataset.jsonl')
        generate_dataset(model_name, num_requests, num_input_tokens, dataset_path)
        return self._run(
            model_name=model_name,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            qps=qps,
            qps_distribution=qps_distribution,
            results_dir=results_dir,
            api_base=api_base,
            api_key=api_key,
            progress_cb=progress_cb,
            log_cb=log_cb,
            dataset_path=dataset_path,
            workload_mode='real_workload',
        )

    def _run(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        num_concurrent_requests: int,
        num_warmup_requests: int = 0,
        qps: Optional[float],
        qps_distribution: str = 'exponential',
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
        dataset_path: str,
        workload_mode: str,
        no_oversample: bool = False,
    ) -> ToolRunResult:
        tokenizer_model_name = get_tokenizer_model_name(model_name)
        base_url = api_base if api_base.endswith('/') else api_base + '/'
        request_rate = str(qps) if qps else 'inf'

        # KEEP IN SYNC WITH ../../vllm/quickstart.sh's VLLM_ARGS.
        args = [
            'vllm',
            'bench',
            'serve',
            '--backend',
            'openai-chat',
            '--base-url',
            base_url,
            '--dataset-name',
            'custom',
            '--dataset-path',
            dataset_path,
            '--endpoint',
            'chat/completions',
            '--model',
            tokenizer_model_name,
            '--served_model_name',
            model_name,
            '--custom-output-len',
            str(num_output_tokens),
            '--num-prompts',
            str(num_requests),
            '--max-concurrency',
            str(num_concurrent_requests),
            '--request-rate',
            request_rate,
            '--burstiness',
            _vllm_burstiness_for_distribution(qps_distribution),
            '--save-detailed',
            '--save-result',
            '--result-dir',
            results_dir,
        ]
        if no_oversample:
            args += ['--no-oversample', '--disable-shuffle']
        if num_warmup_requests > 0:
            args += ['--num-warmups', str(num_warmup_requests)]

        before = set(glob.glob(os.path.join(results_dir, '*.json')))
        progress_cb(0, 1, 'Running vLLM benchmark (this can take a while)')
        # PYTHONUNBUFFERED so vLLM's own stdout is flushed promptly (it's a Python CLI) --
        # otherwise its block-buffering (stdout isn't a tty) delays lines reaching log_cb.
        env = {**os.environ, 'OPENAI_API_KEY': api_key, 'PYTHONUNBUFFERED': '1'}
        self._proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_lines: List[str] = []
        for line in _iter_subprocess_output(self._proc):
            log_lines.append(line)
            if log_cb is not None:
                log_cb(line)
        return_code = self._proc.wait()
        progress_cb(1, 1, 'Running vLLM benchmark (this can take a while)')
        if return_code != 0:
            raise RuntimeError(f'vllm bench serve exited with code {return_code}:\n' + '\n'.join(log_lines[-40:]))

        after = set(glob.glob(os.path.join(results_dir, '*.json')))
        new_files = after - before
        if not new_files:
            raise RuntimeError(f"Could not locate vLLM's native result JSON in {results_dir}.")
        raw_result_path = next(iter(new_files))

        convert_vllm(raw_result_path, model_name, num_input_tokens, num_output_tokens, workload_mode)
        base = raw_result_path[: -len('.json')]
        return ToolRunResult(
            tool='vllm',
            summary_file_path=f'{base}_summary.json',
            individual_responses_file_path=f'{base}_individual_responses.json',
            raw_output_paths=[raw_result_path],
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


class AIPerfToolRunner:
    """Subprocess wrapper around `aiperf profile` -- KEEP IN SYNC WITH ../../aiperf/quickstart.sh."""

    name = 'aiperf'
    display_name = 'aiperf'
    supports_server_metrics = False
    supports_batching_info = False

    def __init__(self) -> None:
        self._proc: Optional['subprocess.Popen[bytes]'] = None

    def run_synthetic(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        qps: Optional[float],
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        # aiperf's own output filenames are fixed within --artifact-dir, so each run needs its own
        # directory (unlike vLLM's timestamped native filenames).
        artifact_dir = os.path.join(results_dir, f'aiperf_{uuid.uuid4().hex[:8]}')
        os.makedirs(artifact_dir, exist_ok=True)
        dataset_path = os.path.join(artifact_dir, 'aiperf_synthetic_dataset.jsonl')
        generate_dataset(model_name, num_requests, num_input_tokens, dataset_path, prompt_key='text')
        return self._run(
            model_name=model_name,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            qps=qps,
            artifact_dir=artifact_dir,
            api_base=api_base,
            api_key=api_key,
            progress_cb=progress_cb,
            log_cb=log_cb,
            dataset_path=dataset_path,
            workload_mode='synthetic',
        )

    def run_custom(
        self,
        *,
        model_name: str,
        uploaded_dataset_path: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        num_output_tokens: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        artifact_dir = os.path.join(results_dir, f'aiperf_{uuid.uuid4().hex[:8]}')
        os.makedirs(artifact_dir, exist_ok=True)

        # aiperf's single_turn custom-dataset-type requires {"text": ...}, not the repo's
        # {"prompt": ...} -- converted here in Python, into this run's own directory (not the
        # jq-into-shared-/tmp approach quickstart.sh uses, which isn't safe for concurrent users).
        text_dataset_path = os.path.join(artifact_dir, 'dataset.jsonl')
        with open(uploaded_dataset_path) as fin, open(text_dataset_path, 'w') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                fout.write(json.dumps({'text': row['prompt']}) + '\n')
        num_requests = _count_jsonl_lines(uploaded_dataset_path)

        return self._run(
            model_name=model_name,
            num_input_tokens=0,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            qps=None,
            artifact_dir=artifact_dir,
            api_base=api_base,
            api_key=api_key,
            progress_cb=progress_cb,
            log_cb=log_cb,
            dataset_path=text_dataset_path,
            workload_mode='custom',
        )

    def run_real_workload(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        qps: float,
        qps_distribution: str,
        num_concurrent_requests: int,
        num_warmup_requests: int,
        timeout: int,
        results_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
    ) -> ToolRunResult:
        artifact_dir = os.path.join(results_dir, f'aiperf_{uuid.uuid4().hex[:8]}')
        os.makedirs(artifact_dir, exist_ok=True)
        dataset_path = os.path.join(artifact_dir, 'aiperf_synthetic_dataset.jsonl')
        generate_dataset(model_name, num_requests, num_input_tokens, dataset_path, prompt_key='text')
        return self._run(
            model_name=model_name,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            num_requests=num_requests,
            num_concurrent_requests=num_concurrent_requests,
            num_warmup_requests=num_warmup_requests,
            qps=qps,
            qps_distribution=qps_distribution,
            artifact_dir=artifact_dir,
            api_base=api_base,
            api_key=api_key,
            progress_cb=progress_cb,
            log_cb=log_cb,
            dataset_path=dataset_path,
            workload_mode='real_workload',
        )

    def _run(
        self,
        *,
        model_name: str,
        num_input_tokens: int,
        num_output_tokens: int,
        num_requests: int,
        num_concurrent_requests: int,
        num_warmup_requests: int = 0,
        qps: Optional[float],
        qps_distribution: str = 'exponential',
        artifact_dir: str,
        api_base: str,
        api_key: str,
        progress_cb: ProgressCallback,
        log_cb: Optional[LogCallback] = None,
        dataset_path: str,
        workload_mode: str,
    ) -> ToolRunResult:
        tokenizer_model_name = get_tokenizer_model_name(model_name)
        request_rate = str(qps) if qps else 'inf'

        # KEEP IN SYNC WITH ../../aiperf/quickstart.sh's `aiperf profile` invocation.
        args = [
            'aiperf',
            'profile',
            '--model',
            model_name,
            '--url',
            api_base,
            '--api-key',
            api_key,
            '--artifact-dir',
            artifact_dir,
            '--endpoint-type',
            'chat',
            '--streaming',
            '--tokenizer',
            tokenizer_model_name,
            '--input-file',
            dataset_path,
            '--custom-dataset-type',
            'single_turn',
            '--request-count',
            str(num_requests),
            '--concurrency',
            str(num_concurrent_requests),
            '--request-rate',
            request_rate,
            '--arrival-pattern',
            _aiperf_arrival_pattern_for_distribution(qps_distribution),
            '--output-tokens-mean',
            str(num_output_tokens),
            # Without this, aiperf auto-detects stdout as a non-tty (true for a subprocess.PIPE)
            # and silently resolves --ui-type to 'none' -- a real no-op UI class
            # (aiperf.ui.no_ui:NoUI) that emits zero progress output, confirmed by invoking
            # aiperf's own CLI-config resolution with these exact args. 'simple' forces its tqdm-
            # based bar instead (same library/behavior as vLLM's, which _iter_subprocess_output
            # already handles correctly).
            '--ui-type',
            'simple',
        ]
        if num_warmup_requests > 0:
            args += ['--warmup-request-count', str(num_warmup_requests)]

        progress_cb(0, 1, 'Running aiperf benchmark (this can take a while)')
        # PYTHONUNBUFFERED so aiperf's own stdout is flushed promptly (it's a Python CLI) --
        # otherwise its block-buffering (stdout isn't a tty) delays lines reaching log_cb.
        env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
        self._proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_lines: List[str] = []
        for line in _iter_subprocess_output(self._proc):
            log_lines.append(line)
            if log_cb is not None:
                log_cb(line)
        return_code = self._proc.wait()
        progress_cb(1, 1, 'Running aiperf benchmark (this can take a while)')
        if return_code != 0:
            raise RuntimeError(f'aiperf profile exited with code {return_code}:\n' + '\n'.join(log_lines[-40:]))

        convert_aiperf(
            artifact_dir,
            model_name,
            workload_mode,
            num_concurrent_requests,
            qps,
            num_input_tokens,
            num_output_tokens,
            None,
        )
        return ToolRunResult(
            tool='aiperf',
            summary_file_path=os.path.join(artifact_dir, 'aiperf_summary.json'),
            individual_responses_file_path=os.path.join(artifact_dir, 'aiperf_individual_responses.json'),
            raw_output_paths=[
                os.path.join(artifact_dir, 'profile_export.jsonl'),
                os.path.join(artifact_dir, 'profile_export_aiperf.json'),
            ],
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


TOOL_RUNNERS: Dict[str, ToolRunner] = {
    'kit': KitToolRunner(),
    'vllm': VLLMToolRunner(),
    'aiperf': AIPerfToolRunner(),
}
