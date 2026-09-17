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
    create_log_callback,
    create_progress_callback,
    model_selector_widget,
    render_logo,
    render_multi_tool_results,
    render_title_icon,
    save_uploaded_file,
    set_api_variables,
    set_font,
    setup_credentials,
)
from benchmarking.utils import SAMBANOVA_API_BASE

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

CONFIG_PATH = './config.yaml'
with open(CONFIG_PATH) as file:
    st.session_state.config = yaml.safe_load(file)
    st.session_state.prod_mode = st.session_state.config['prod_mode']
    st.session_state.pages_to_show = st.session_state.config['pages_to_show']


def _initialize_sesion_variables() -> None:
    # Clear results and reset the tool selector to Kit-only when navigating from a different page
    # -- otherwise a multi-tool selection made on another eval page would silently carry over here.
    if st.session_state.get('current_page') != 'custom':
        st.session_state.tool_results = None
        st.session_state.current_page = 'custom'
        st.session_state.selected_tools = ['kit']
        st.session_state.previous_selected_tools = ['kit']

    # Initialize llm
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'llm_api' not in st.session_state:
        st.session_state.llm_api = None

    # Initialize llm params
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'file_path' not in st.session_state:
        st.session_state.file_path = None
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = None

    # Tool selection -- which benchmarking tool(s) to run this pass. Kit-only by default;
    # selecting more than one triggers the N-way comparison section.
    if 'selected_tools' not in st.session_state:
        st.session_state.selected_tools = ['kit']
    if 'previous_selected_tools' not in st.session_state:
        st.session_state.previous_selected_tools = st.session_state.selected_tools

    # Additional initialization
    if 'run_button' in st.session_state and st.session_state.run_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False
    if 'tool_results' not in st.session_state:
        st.session_state.tool_results = None
    if 'tool_logs' not in st.session_state:
        st.session_state.tool_logs = {}
    if 'failed_tools' not in st.session_state:
        st.session_state.failed_tools = []
    if 'zip_buffer' not in st.session_state:
        st.session_state.zip_buffer = None
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

    render_title_icon('Custom Performance Evaluation', os.path.join(repo_dir, 'images', 'benchmark_icon.png'))
    st.markdown(
        'Here you can select a custom dataset that you want to benchmark performance with. Note that with models that \
          support dynamic batching, you are limited to the number of cpus available on your machine to send concurrent \
              requests.'
    )

    with st.sidebar:
        # Set up credentials and API variables
        setup_credentials()

        render_logo()
        ##################
        # File Selection #
        ##################
        st.title('File Selection')
        st.session_state.uploaded_file = st.file_uploader(
            'Upload JSON File', type='jsonl', disabled=st.session_state.running
        )
        st.session_state.file_path = save_uploaded_file(internal_save_path='data/custom_input_files')

        #########################
        # Runtime Configuration #
        #########################
        st.title('Configuration')

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

        st.session_state.llm = model_selector_widget(disabled=st.session_state.running)

        if st.session_state.llm_api == 'sncloud':
            st.selectbox(
                'API type',
                options=list(LLM_API_OPTIONS.keys()),
                format_func=lambda x: LLM_API_OPTIONS[x],
                index=0,
                disabled=True,
            )

        st.number_input(
            'Number of output tokens',
            min_value=1,
            max_value=2048,
            value=256,
            step=1,
            key='output_tokens',
            disabled=st.session_state.running,
            help='Caps generated tokens per request. Same field as the synthetic/real-workload '
            'pages, just optional here since the custom dataset already sets input length.',
        )

        st.number_input(
            'Num Concurrent Requests',
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key='number_concurrent_requests',
            disabled=st.session_state.running,
        )

        non_kit_selected = any(t != 'kit' for t in st.session_state.selected_tools)
        st.number_input(
            'Num Warm-up Requests',
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key='number_warmup_requests',
            disabled=st.session_state.running,
            help='Throwaway requests sent (at the concurrency above) before the measured run. Their results '
            'are discarded so cold-start and batch ramp-up costs do not skew the metrics. 0 disables warm-up. ',
        )

        st.number_input(
            'Timeout',
            min_value=60,
            max_value=1800,
            value=600,
            step=1,
            key='timeout',
            disabled=st.session_state.running or non_kit_selected,
            help='Number of seconds before program times out. Not supported by vLLM/aiperf.',
        )
        if non_kit_selected:
            st.caption('ℹ️ Timeout is not supported by vLLM/aiperf.')

        job_submitted = st.sidebar.button(
            'Run!',
            disabled=st.session_state.running or not st.session_state.selected_tools,
            key='run_button',
            type='primary',
            width='stretch',
        )

        sidebar_stop = st.sidebar.button(
            'Stop', disabled=not st.session_state.running, type='secondary', width='stretch'
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

    if job_submitted:
        st.session_state.mp_events.input_submitted('custom_performance_evaluation ')
        st.toast(
            """Performance evaluation in progress. This could take a while depending on the dataset size and max tokens
              setting."""
        )
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
                failed_tools: List[str] = []
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
                            results[tool_name] = runner.run_custom(
                                model_name=st.session_state.llm,
                                uploaded_dataset_path=st.session_state.file_path,
                                num_concurrent_requests=st.session_state.number_concurrent_requests,
                                num_warmup_requests=st.session_state.number_warmup_requests,
                                num_output_tokens=st.session_state.output_tokens,
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
                            failed_tools.append(tool_name)
                        finally:
                            st.session_state.tool_logs[tool_name] = log_lines

                # Committed once the benchmark runs themselves succeed -- a failure building the
                # download zip below must not blank out already-successful results.
                st.session_state.tool_results = results
                st.session_state.failed_tools = failed_tools
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
            st.session_state.tool_results,
            st.session_state.output_tokens,
            st.session_state.tool_logs,
            st.session_state.failed_tools,
        )


if __name__ == '__main__':
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon=os.path.join(repo_dir, 'images', 'icon.svg'),
    )

    _initialize_sesion_variables()

    main()
