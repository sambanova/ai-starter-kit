import io
import os
import warnings
import zipfile
from typing import Dict, List

import streamlit as st
import yaml

from benchmarking.benchmarking_tools.kit.src.tool_runners import TOOL_RUNNERS, ToolRunResult
from benchmarking.streamlit.streamlit_utils import (
    LLM_API_OPTIONS,
    MULTIMODAL_IMAGE_SIZE_OPTIONS,
    create_log_callback,
    create_progress_callback,
    model_selector_widget,
    render_multi_tool_results,
    render_logo,
    render_title_icon,
    set_api_variables,
    set_font,
    setup_credentials,
)
from benchmarking.utils import CONFIG_PATH, SAMBANOVA_API_BASE

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']


def _initialize_session_variables() -> None:
    # Clear results and reset the tool selector to Kit-only when navigating from a different page
    # -- otherwise a multi-tool selection made on another eval page would silently carry over here.
    if st.session_state.get('current_page') != 'synthetic':
        st.session_state.tool_results = None
        st.session_state.zip_buffer = None
        st.session_state.current_page = 'synthetic'
        st.session_state.selected_tools = ['kit']
        st.session_state.previous_selected_tools = ['kit']

    # Initialize llm
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    # Initialize llm params
    if 'multimodal_image_size' not in st.session_state:
        st.session_state.multimodal_image_size = None
    if 'input_tokens' not in st.session_state:
        st.session_state.input_tokens = None
    if 'output_tokens' not in st.session_state:
        st.session_state.output_tokens = None
    if 'number_requests' not in st.session_state:
        st.session_state.number_requests = None
    if 'number_concurrent_requests' not in st.session_state:
        st.session_state.number_concurrent_requests = None
    if 'number_warmup_requests' not in st.session_state:
        st.session_state.number_warmup_requests = None
    if 'timeout' not in st.session_state:
        st.session_state.timeout = None
    if 'llm_api' not in st.session_state:
        st.session_state.llm_api = None

    # Tool selection -- which benchmarking tool(s) to run this pass. Kit-only by default;
    # selecting more than one triggers the N-way comparison section.
    if 'selected_tools' not in st.session_state:
        st.session_state.selected_tools = ['kit']
    if 'previous_selected_tools' not in st.session_state:
        st.session_state.previous_selected_tools = st.session_state.selected_tools

    # Additional initializations
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'run_button' in st.session_state and st.session_state.run_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False
    if 'zip_buffer' not in st.session_state:
        st.session_state.zip_buffer = None
    if 'tool_results' not in st.session_state:
        st.session_state.tool_results = None
    if 'tool_logs' not in st.session_state:
        st.session_state.tool_logs = {}
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None
    if 'mp_events' not in st.session_state:
        st.switch_page('app.py')


def _build_download_zip(results: Dict[str, ToolRunResult]) -> io.BytesIO:
    """Zips every tool's standardized + native output files for download, one subfolder per tool
    that ran (e.g. 'Kit/', 'vLLM/', 'aiperf/') so multi-tool downloads stay organized.

    Copies each file's raw bytes as-is (no JSON parse/re-serialize) -- some raw outputs (e.g.
    aiperf's profile_export.jsonl) are JSONL (one JSON object per line), not a single JSON
    document, so `json.loads(f.read())` would fail with "Extra data" on line 2.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for tool_name, result in results.items():
            folder = TOOL_RUNNERS[tool_name].display_name
            for file_path in [
                result.summary_file_path,
                result.individual_responses_file_path,
                *result.raw_output_paths,
            ]:
                zip_file.write(file_path, arcname=f'{folder}/{os.path.basename(file_path)}')
    zip_buffer.seek(0)
    return zip_buffer


def main() -> None:
    set_font()

    render_title_icon('Synthetic Performance Evaluation', os.path.join(repo_dir, 'images', 'benchmark_icon.png'))

    st.markdown(
        """This performance evaluation assesses the following LLM's performance metrics using concurrent processes.
        _client represents the metrics computed from the client-side (includes queue and round-trip time
        from host to server and back) and _server represents the metrics computed from the server-side."""
    )
    st.markdown(
        """**Time to first token (TTFT):** This metric is driven by the time required to process the prompt and then
        generate the first output token."""
    )
    st.markdown('**E2E Latency:** TTFT + (Time per Output Token) * (the number of tokens to be generated - 1)')
    st.markdown(
        """**Tokens/sec/request (Output Throughput)**: Number of output tokens generated per second per request
        for a given batch-size. Client metric is calculated as *Number of Output Tokens / (E2E Latency - TTFT)*"""
    )
    st.markdown("""**Tokens/sec (Throughput)**: Total number of tokens generated per second for a given batch-size.""")

    with st.sidebar:
        # Set up credentials and API variables
        setup_credentials()

        render_logo()
        st.title('Configuration')
        st.markdown('**Modify the following parameters before running the process**')

        st.multiselect(
            'Benchmark tool(s) to run',
            options=list(TOOL_RUNNERS.keys()),
            format_func=lambda t: TOOL_RUNNERS[t].display_name,
            help='Select one or more tools to run this pass. Selecting more than one shows a '
            'side-by-side comparison once all selected tools finish.',
            disabled=st.session_state.running,
            key='selected_tools',
        )
        if not st.session_state.selected_tools:
            st.warning('Select at least one benchmark tool to run.')

        if st.session_state.selected_tools != st.session_state.previous_selected_tools:
            st.session_state.tool_results = None
            st.session_state.zip_buffer = None
            st.session_state.previous_selected_tools = st.session_state.selected_tools

        st.divider()

        st.session_state.llm = model_selector_widget(
            disabled=st.session_state.running
        )

        if st.session_state.llm_api == 'sncloud':
            st.selectbox(
                'API type',
                options=list(LLM_API_OPTIONS.keys()),
                format_func=lambda x: LLM_API_OPTIONS[x],
                index=0,
                disabled=True,
            )

        # Multimodal image size and a configurable timeout are Kit-only concepts -- forced to
        # their defaults whenever any non-Kit tool is selected, since vLLM/aiperf runs don't
        # support either.
        non_kit_selected = any(t != 'kit' for t in st.session_state.selected_tools)
        st.session_state.multimodal_image_size = st.selectbox(
            'Multimodal image size',
            options=list(MULTIMODAL_IMAGE_SIZE_OPTIONS.keys()),
            format_func=lambda x: MULTIMODAL_IMAGE_SIZE_OPTIONS[x],
            index=0,
            disabled=st.session_state.running or non_kit_selected,
            help='Select the pre-set image size for multimodal models. '
            'Small: 500x500, Medium: 1024x1024, Large: 2000x2000. Select N/A for non-multimodal models. '
            'Not supported by vLLM/aiperf.',
        )
        if non_kit_selected:
            st.session_state.multimodal_image_size = 'na'
            st.caption('ℹ️ Multimodal image size is not supported by vLLM/aiperf.')

        st.session_state.input_tokens = st.number_input(
            'Number of input tokens',
            min_value=50,
            max_value=10000,
            value=1000,
            step=1,
            disabled=st.session_state.running,
        )

        st.session_state.output_tokens = st.number_input(
            'Number of output tokens',
            min_value=50,
            max_value=2000,
            value=1000,
            step=1,
            disabled=st.session_state.running,
        )

        st.session_state.number_requests = st.number_input(
            'Number of total requests',
            min_value=1,
            max_value=2000,
            value=10,
            step=1,
            disabled=st.session_state.running,
        )

        st.session_state.number_concurrent_requests = st.number_input(
            'Number of concurrent requests',
            min_value=1,
            max_value=2000,
            value=1,
            step=1,
            disabled=st.session_state.running,
        )

        st.session_state.number_warmup_requests = st.number_input(
            'Number of warm-up requests',
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            disabled=st.session_state.running,
            help='Throwaway requests sent (at the concurrency above) before the measured run. Their results '
            'are discarded so cold-start and batch ramp-up costs do not skew the metrics. 0 disables warm-up.',
        )

        st.session_state.timeout = st.number_input(
            'Timeout',
            min_value=60,
            max_value=1800,
            value=600,
            step=1,
            disabled=st.session_state.running or non_kit_selected,
            help='Number of seconds before program times out. Not supported by vLLM/aiperf.',
        )
        if non_kit_selected:
            st.caption('ℹ️ Timeout is not supported by vLLM/aiperf.')

        st.session_state.running = st.sidebar.button(
            'Run!',
            disabled=st.session_state.running or not st.session_state.selected_tools,
            key='run_button',
            type='primary',
            width='stretch',
        )

        # Stop only ever means "abort a run in progress" -- it's disabled whenever nothing is
        # running, and never gates Run!, so a finished run doesn't require a Stop press just to
        # be able to start another one.
        sidebar_stop = st.sidebar.button(
            'Stop',
            disabled=not st.session_state.running,
            type='secondary',
            width='stretch',
        )

        # Always rendered in the same spot (so it doesn't pop in/out of the sidebar layout) --
        # disabled until there's a zip ready AND disabled again once a new run starts, so it can
        # never be clicked mid-run against a stale zip from a previous pass.
        st.sidebar.download_button(
            label='Download Results',
            data=st.session_state.zip_buffer if st.session_state.zip_buffer is not None else b'',
            file_name='output_files.zip',
            mime='application/zip',
            disabled=st.session_state.running or st.session_state.zip_buffer is None,
            width='stretch',
        )

    if sidebar_stop:
        st.session_state.running = False
        for runner in TOOL_RUNNERS.values():
            runner.stop()

    if st.session_state.running:
        st.session_state.mp_events.input_submitted('synthetic_performance_evaluation ')
        tool_labels = ', '.join(TOOL_RUNNERS[t].display_name for t in st.session_state.selected_tools)
        st.toast(f'{tool_labels} performance evaluation processing now. It should take a few minutes.')
        with st.spinner('Processing'):
            do_rerun = False
            try:
                # set_api_variables() returns {} outside prod_mode -- fall back to os.environ
                # (populated by app.py's load_dotenv) exactly like the Kit's own HTTP client does
                # internally (SambaNovaClient falls back to os.environ when api_variables is
                # empty), so vLLM/aiperf's subprocess runners see the same credentials Kit does.
                api_variables = set_api_variables()
                api_base = api_variables.get('SAMBANOVA_API_BASE') or os.environ.get(
                    'SAMBANOVA_API_BASE', SAMBANOVA_API_BASE
                )
                api_key = api_variables.get('SAMBANOVA_API_KEY') or os.environ.get('SAMBANOVA_API_KEY', '')
                results: Dict[str, ToolRunResult] = {}
                st.session_state.tool_logs = {}
                for tool_name in st.session_state.selected_tools:
                    runner = TOOL_RUNNERS[tool_name]
                    # Each tool gets its own status container: a standardized progress bar + log
                    # area, decoupled from every other tool's (no shared/reused widgets), that
                    # auto-collapses once that tool's run finishes so a multi-tool pass doesn't
                    # pile up walls of log text once everything's done. The log lines themselves
                    # are also persisted into session_state (see render_tool_logs) so they remain
                    # available -- collapsed -- once results are shown after this run's rerun.
                    with st.status(f'{runner.display_name}: running...', expanded=True) as status:
                        progress_bar = st.progress(0.0)
                        log_placeholder = st.empty()
                        log_lines: List[str] = []
                        try:
                            results[tool_name] = runner.run_synthetic(
                                model_name=st.session_state.llm,
                                num_input_tokens=st.session_state.input_tokens,
                                num_output_tokens=st.session_state.output_tokens,
                                num_requests=st.session_state.number_requests,
                                num_concurrent_requests=st.session_state.number_concurrent_requests,
                                num_warmup_requests=st.session_state.number_warmup_requests,
                                qps=None,
                                timeout=st.session_state.timeout,
                                results_dir=f'./data/results/{tool_name}',
                                api_base=api_base,
                                api_key=api_key,
                                progress_cb=create_progress_callback(progress_bar),
                                log_cb=create_log_callback(log_placeholder, log_lines, runner.display_name),
                            )
                            status.update(label=f'{runner.display_name}: done', state='complete', expanded=False)
                        except Exception as e:
                            status.update(label=f'{runner.display_name}: failed', state='error', expanded=True)
                            st.error(f'{runner.display_name} failed: {e}')
                        finally:
                            st.session_state.tool_logs[tool_name] = log_lines

                # Committed once the benchmark runs themselves succeed -- a failure building the
                # download zip below must not blank out already-successful results.
                st.session_state.tool_results = results
                st.session_state.running = False

                try:
                    st.session_state.zip_buffer = _build_download_zip(results)
                except Exception as zip_error:
                    st.error(f'Could not build the download zip: {zip_error}')
                    st.session_state.zip_buffer = None

                # workareound to avoid rerun within try block
                do_rerun = True
            except Exception as e:
                st.error(f'Error:\n{e}.')
                st.session_state.tool_results = None
            if do_rerun:
                st.rerun()

    if st.session_state.tool_results:
        render_multi_tool_results(
            st.session_state.tool_results, st.session_state.output_tokens, st.session_state.tool_logs
        )


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'icon.svg'),
    )

    _initialize_session_variables()

    main()
