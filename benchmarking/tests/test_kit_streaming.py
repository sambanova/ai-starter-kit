"""Unit tests for how the Kit's client reads a streamed chat completion.

Each test feeds `SambaNovaCloudAPI.compute_metrics` a fake SSE stream (no network, no API key)
with a fake clock, so the timings are exact. Run from the repo root:

    python benchmarking/tests/test_kit_streaming.py
    # or
    python -m unittest benchmarking.tests.test_kit_streaming -v
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest import mock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarking.benchmarking_tools.kit.src.llmperf import common_metrics  # noqa: E402
from benchmarking.benchmarking_tools.kit.src.llmperf import sambanova_client as client  # noqa: E402
from benchmarking.benchmarking_tools.kit.src.llmperf.models import RequestConfig  # noqa: E402

# A stream event: (seconds since the request started, chunk payload).
Event = Tuple[float, Dict[str, Any]]


class WordTokenizer:
    """Counts words as tokens, so token counts in the tests are easy to read."""

    def encode(self, text: str) -> List[str]:
        return text.split()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def chunk(
    content: Optional[str] = None,
    reasoning: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {'role': 'assistant'}
    for key, value in (('content', content), ('reasoning', reasoning), ('reasoning_content', reasoning_content)):
        if value is not None:
            delta[key] = value
    payload: Dict[str, Any] = {'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]}
    if usage is not None:
        payload['usage'] = usage
    return payload


def usage(completion_tokens: int, **extra: Any) -> Dict[str, Any]:
    return {
        'prompt_tokens': 10,
        'completion_tokens': completion_tokens,
        'total_tokens': 10 + completion_tokens,
        **extra,
    }


def make_api(sampling_params: Optional[Dict[str, Any]] = None) -> client.SambaNovaCloudAPI:
    config = RequestConfig(
        request_idx=0,
        model='test-model',
        prompt_tuple=({'name': 'custom_prompt', 'template': 'hello'}, 10),
        sampling_params=dict(sampling_params or {'max_tokens_to_generate': 100}),
        llm_api='sncloud',
        api_variables={'SAMBANOVA_API_BASE': 'https://example.invalid/v1', 'SAMBANOVA_API_KEY': 'test'},
        is_stream_mode=True,
    )
    return client.SambaNovaCloudAPI(config, WordTokenizer())


def run_stream(events: List[Event], end_at: Optional[float] = None) -> Tuple[Dict[str, Any], str]:
    """Runs compute_metrics over `events`; the stream closes at `end_at` (default: last event)."""
    clock = FakeClock()
    start = clock.now

    def sse_events() -> Iterator[SimpleNamespace]:
        for at, payload in events:
            clock.now = start + at
            yield SimpleNamespace(data=json.dumps(payload))
        clock.now = start + (end_at if end_at is not None else (events[-1][0] if events else 0.0))
        yield SimpleNamespace(data='[DONE]')

    response = mock.MagicMock(status_code=200)
    response.__enter__.return_value = response
    with (
        mock.patch(f'{client.__name__}.requests.post', return_value=response),
        mock.patch(f'{client.__name__}.sseclient.SSEClient', return_value=SimpleNamespace(events=sse_events)),
        mock.patch(f'{client.__name__}.time', SimpleNamespace(monotonic=clock.monotonic)),
    ):
        return make_api().compute_metrics({})


class UsageOnEveryChunk(unittest.TestCase):
    """Fix 1: text is read from every chunk with a delta, whether or not it carries `usage`."""

    def test_text_is_read_when_every_chunk_carries_usage(self) -> None:
        # Fireworks-style: usage on every chunk, growing as the reply is generated.
        events = [
            (0.10, chunk(content='', usage=usage(0))),
            (0.40, chunk(content='Hi', usage=usage(1))),
            (0.50, chunk(content=' there friend', usage=usage(3))),
            (0.60, chunk(content=' how are', usage=usage(5))),
            (0.70, chunk(content=' you', usage=usage(6), finish_reason='stop')),
        ]
        metrics, text = run_stream(events)
        self.assertEqual(text, 'Hi there friend how are you')
        self.assertAlmostEqual(metrics[common_metrics.TTFT], 0.40)
        self.assertAlmostEqual(metrics[common_metrics.E2E_LAT], 0.70)
        # The last usage seen is the complete one.
        self.assertEqual(metrics[common_metrics.NUM_OUTPUT_TOKENS], 6)
        self.assertAlmostEqual(metrics[common_metrics.REQ_OUTPUT_THROUGHPUT], 6 / (0.70 - 0.40))

    def test_usage_only_on_a_final_chunk_without_choices(self) -> None:
        # OpenAI-style: usage arrives on its own chunk with `choices: []`.
        events = [
            (0.20, chunk(content='one')),
            (0.30, chunk(content=' two')),
            (0.40, chunk(content=' three', finish_reason='stop')),
            (0.45, {'choices': [], 'usage': usage(3)}),
        ]
        metrics, text = run_stream(events)
        self.assertEqual(text, 'one two three')
        self.assertEqual(metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER], 3)

    def test_stream_without_any_usage(self) -> None:
        metrics, text = run_stream([(0.2, chunk(content='a')), (0.3, chunk(content=' b')), (0.4, chunk(content=' c'))])
        self.assertEqual(text, 'a b c')
        self.assertIsNone(metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER])
        self.assertEqual(metrics[common_metrics.NUM_OUTPUT_TOKENS], 3)  # from the client tokenizer


class ReasoningFields(unittest.TestCase):
    """Fix 2: both `reasoning` and `reasoning_content` count as generated text."""

    def check_reasoning_field(self, field: str) -> None:
        def thought(text: str) -> Dict[str, Any]:
            return chunk(reasoning=text) if field == 'reasoning' else chunk(reasoning_content=text)

        events = [
            (0.25, thought('think')),
            (0.35, thought(' more')),
            (0.60, chunk(content=' answer', usage=None)),
            (0.70, chunk(content=' done', finish_reason='stop')),
        ]
        metrics, text = run_stream(events)
        self.assertEqual(text, 'think more answer done')
        # First token is the first reasoning token, not the first answer word.
        self.assertAlmostEqual(metrics[common_metrics.TTFT], 0.25)

    def test_reasoning(self) -> None:
        self.check_reasoning_field('reasoning')

    def test_reasoning_content(self) -> None:
        self.check_reasoning_field('reasoning_content')

    def test_delta_with_reasoning_and_content_keeps_both(self) -> None:
        self.assertEqual(client.delta_text({'reasoning_content': 'why ', 'content': 'what'}), 'why what')
        self.assertEqual(client.delta_text({'role': 'assistant', 'content': None}), '')


class EmptyChunks(unittest.TestCase):
    """Fix 3: chunks with empty text don't count as tokens for first-token timing."""

    def test_empty_content_does_not_start_the_ttft_clock(self) -> None:
        events = [
            (0.05, chunk(content='')),  # role-only first chunk
            (0.10, chunk(reasoning='')),
            (0.50, chunk(content='first')),
            (0.60, chunk(content=' second')),
            (0.70, chunk(content=' third', finish_reason='stop')),
        ]
        metrics, _ = run_stream(events)
        self.assertAlmostEqual(metrics[common_metrics.TTFT], 0.50)
        self.assertEqual(len(metrics[common_metrics.INTER_TOKEN_LATENCY]), 3)


class IgnoreEos(unittest.TestCase):
    """Fix 4: ignore_eos stays on by default but can be turned off in sampling params."""

    def test_default_is_true(self) -> None:
        self.assertIs(make_api()._get_json_data()['ignore_eos'], True)

    def test_can_be_turned_off(self) -> None:
        body = make_api({'max_tokens_to_generate': 100, 'ignore_eos': False})._get_json_data()
        self.assertIs(body['ignore_eos'], False)
        self.assertEqual(body['max_tokens'], 100)


class ChunkyStreams(unittest.TestCase):
    """Fix 5: client tokens/sec doesn't blow up when text arrives in a few near-simultaneous chunks."""

    def test_burst_falls_back_to_server_figure(self) -> None:
        # Whole reply in three chunks within 0.2 ms: the old formula reported ~150,000 tokens/s.
        server = usage(60, completion_tokens_after_first_per_sec=600.0)
        events = [
            (0.5000, chunk(content='w ' * 20)),
            (0.5001, chunk(content='w ' * 20)),
            (0.5002, chunk(content='w ' * 20, finish_reason='stop')),
            (0.5003, {'choices': [], 'usage': server}),
        ]
        metrics, _ = run_stream(events)
        self.assertEqual(metrics[common_metrics.REQ_OUTPUT_THROUGHPUT], 600.0)
        self.assertEqual(metrics['extra']['client_output_token_per_s_source'], 'server')

    def test_burst_without_server_figure_reports_none(self) -> None:
        events = [(0.5, chunk(content='a b')), (0.5001, chunk(content=' c d')), (0.5002, chunk(content=' e'))]
        metrics, _ = run_stream(events)
        self.assertIsNone(metrics[common_metrics.REQ_OUTPUT_THROUGHPUT])
        self.assertIsNone(metrics['extra']['client_output_token_per_s_source'])

    def test_two_chunks_are_too_few_to_time(self) -> None:
        # Two chunks, the stream closing 1.2 s later: the old formula reported ~16 tokens/s.
        events = [(0.3, chunk(content='a ' * 10)), (0.31, chunk(content='b ' * 10))]
        metrics, _ = run_stream(events, end_at=1.5)
        self.assertIsNone(metrics[common_metrics.REQ_OUTPUT_THROUGHPUT])

    def test_well_spread_stream_is_timed_client_side_as_before(self) -> None:
        events = [(0.2 + 0.01 * i, chunk(content=f' t{i}')) for i in range(50)]
        metrics, _ = run_stream(events)
        expected = 50 / (metrics[common_metrics.E2E_LAT] - metrics[common_metrics.TTFT])
        self.assertAlmostEqual(metrics[common_metrics.REQ_OUTPUT_THROUGHPUT], expected)
        self.assertEqual(metrics['extra']['client_output_token_per_s_source'], 'client')

    def test_single_chunk_keeps_end_to_end_rate(self) -> None:
        metrics, _ = run_stream([(0.4, chunk(content='a b c d', finish_reason='stop'))])
        self.assertAlmostEqual(metrics[common_metrics.REQ_OUTPUT_THROUGHPUT], 4 / 0.4)
        self.assertEqual(metrics['extra']['client_output_token_per_s_source'], 'client_end_to_end')


if __name__ == '__main__':
    unittest.main(verbosity=2)
