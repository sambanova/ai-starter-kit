import subprocess
import datetime
from collections import defaultdict
import threading
import time
import yaml
import random
import pandas as pd
import re
import argparse
import requests
import json
import sys
import pprint
sys.path.append('../../') # For running from sambaStudio/SambaX
sys.path.append('Infra') # For running from home
sys.path.append('sambaStudio') # For running from home
from generate_prompts import Prompt

def clean_string(input_string):
    # Use regex to match the desired pattern and extract it
    match = re.search(r'\b[A-Za-z0-9]+\b', input_string)
    if match:
        return match.group(0)
    else:
        return ""

def read_master_menu(filepath):
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Convert DataFrame to dictionary with 'model_name' as the key
    model_dict = defaultdict(dict)
    for index, row in df.iterrows():
        model_name = row['model_name']
        app_name = row['app_name']
        if pd.notna(model_name):  # Check if model_name is not NaN
            model_name = app_name.strip() + '+' + model_name.strip()  # Strip whitespace only if it's a string
            # Create a dictionary for the current row with specific transformations
            model_details = {}
            for column, value in row.items():
                if pd.notna(value):
                    value = str(value).strip()
                    if column == 'Param Count':
                        # Convert float string to int correctly
                        model_details[column] = f"{int(float(value))}b"  # Format with 'b' suffix
                    elif column == '3-graph':
                        # Process '3-graph' entries
                        model_details[column] = [int(float(x.replace('k', '')) * 1024) if 'k' in x else int(float(x)) for x in value.split(',')]
                    elif column == 'Batch Sizes':
                        # Convert batch sizes to integer list
                        model_details[column] = [int(float(x)) for x in value.split(',')]
                    else:
                        model_details[column] = value  # Keep other fields as strings

            model_dict[model_name] = model_details

    # Remove the 'model_name' key from each sub-dictionary
    for model in list(model_dict):
        if 'model_name' in model_dict[model]:
            del model_dict[model]['model_name']
        
        if 'BYOC candidate' not in model_dict[model]:
            model_dict[model]['BYOC candidate'] = None 

    return model_dict

def extract_triviaqa_prompts(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    res_dict = {
        'doc_id': [],
        'prompt': [],
        'rdu_o': [],
        'golden': [],
        'score': []
    }

    for item in data:
        res_dict['doc_id'].append(item['doc_id'])
        res_dict['prompt'].append(item['prompt_0'].replace('Answer:until', 'Answer:').strip()) #.replace('\n', '\\n'))
        res_dict['rdu_o'].append(item['logit_0'].strip()) #.replace('\n', '\\n'))
        res_dict['golden'].append(item['truth'].strip()) #.replace('\n', '\\n'))
        try:
            res_dict['score'].append(item['em'])
        except:
            res_dict['score'].append(item['acc'])

    return res_dict

def parse_curl_response(lines, streamflag):

    # assert isinstance(lines, list) and len(lines) > 0

    completion = None
    tokenscnt = None
    sidenote = None
    app_total_latency = None
    app_total_tokens_per_sec = None
    app_completion_tokens_after_first_per_sec_first_ten = None
    app_prompt_tokens_count = None
    app_total_tokens_count = None
    app_completion_tokens_after_first_per_sec = None
    app_completion_tokens_per_sec = None
    app_time_to_first_token = None
    app_batch_size_used = None

    data = json.loads(lines)
    # print(data)

    ## for now, just separate stream and completion
    if streamflag == True:
        try:
            data = data['result']['items'][0]['value']
        except KeyError:
            return 'TimeoutLong', data, 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'

        try:
            app_total_tokens_per_sec = data['total_tokens_per_sec']
        except:
            pass

        try:
            app_batch_size_used = data['batch_size_used']
        except:
            pass

        try:
            app_total_latency = data['model_execution_time']
        except:
            pass

        try:
            app_completion_tokens_after_first_per_sec_first_ten = data['completion_tokens_after_first_per_sec_first_ten']
        except:
            pass

        try:
            app_total_tokens_count = data['total_tokens_count']
        except:
            pass

        try:
            app_prompt_tokens_count = data['prompt_tokens_count']
        except:
            pass

        try:
            app_completion_tokens_after_first_per_sec = data['throughput_after_first_token']
        except:
            pass

        try:
            app_completion_tokens_per_sec = data['completion_tokens_per_sec']
        except:
            pass

        try:
            tokenscnt = data['total_tokens_count'] - data['prompt_tokens_count']
        except (KeyError, ValueError, TypeError):
            try:
                tokenscnt = len(data['tokens'])
            except KeyError:
                tokenscnt = None
                sidenote = 'Error'

        try:
            app_time_to_first_token = data['time_to_first_token']
        except:
            pass

        try:
            completion = data['completion']
        except:
            pass

        return completion, tokenscnt, sidenote, app_total_latency, app_total_tokens_per_sec, app_completion_tokens_after_first_per_sec_first_ten, app_total_tokens_count, app_prompt_tokens_count, app_completion_tokens_after_first_per_sec, app_completion_tokens_per_sec, app_time_to_first_token, app_batch_size_used
    else:
        completion = data['items'][0]['value']
        try:
            tokenscnt = len(data['items'][0]['value']['tokens'])
        except:
            tokenscnt = len(data['items'][0]['value'])

        try:
            return completion['completion'], tokenscnt, 'Prompt Mode', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'
        except:
            return completion, tokenscnt, 'Prompt Mode', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'
        

def build_prompt_data_part(maxtoken, prompt, expert, dosample, processprompt, streamflag):
    maxtok = maxtoken
    penalty = 1
    temperature = 0.01
    topk = 10
    toplog = 0
    topp = 0.95

    if expert == 'standalone-model':
        if streamflag == True:
            datapart = r'''{"instance":"''' + str(prompt) + \
                        r'''","params":{"do_sample":{"type":"bool","value":"''' + str(dosample) + \
                        r'''"},"max_tokens_to_generate":{"type":"int","value":"''' + str(maxtok) + \
                        r'''"},"repetition_penalty":{"type":"float","value":"''' + str(penalty) + \
                        r'''"},"temperature":{"type":"float","value":"''' + str(temperature) + \
                        r'''"},"top_k":{"type":"int","value":"''' + str(topk) + \
                        r'''"},"top_logprobs":{"type":"int","value":"''' + str(toplog) + \
                        r'''"},"top_p":{"type":"float","value":"''' + str(topp) + \
                        r'''"}}}'''
        else:
            datapart = r'''{"instances":["''' + str(prompt) + \
                        r'''"],"params":{"do_sample":{"type":"bool","value":"''' + str(dosample) + \
                        r'''"},"max_tokens_to_generate":{"type":"int","value":"''' + str(maxtok) + \
                        r'''"},"repetition_penalty":{"type":"float","value":"''' + str(penalty) + \
                        r'''"},"temperature":{"type":"float","value":"''' + str(temperature) + \
                        r'''"},"top_k":{"type":"int","value":"''' + str(topk) + \
                        r'''"},"top_logprobs":{"type":"int","value":"''' + str(toplog) + \
                        r'''"},"top_p":{"type":"float","value":"''' + str(topp) + \
                        r'''"}}}'''
    else:  
        if processprompt == 'false':
            if streamflag == True:
                datapart = r'''{"items":[{"id":"item1", "value":"''' + str(prompt) + \
                            r'''"}],"params":{"do_sample":''' + str(dosample) + \
                            r''',"max_tokens_to_generate":''' + str(maxtok) + \
                            r''',"process_prompt":''' + str(processprompt) + \
                            r''',"repetition_penalty":''' + str(penalty) + \
                            r''',"select_expert":"''' + str(expert) + \
                            r'''","temperature":''' + str(temperature) + \
                            r''',"top_k":''' + str(topk) + \
                            r''',"top_p":''' + str(topp) + \
                            r'''}}'''
            else:
                datapart = r'''{"items":[{"id":"item1", "value":"''' + str(prompt) + \
                            r'''"}],"params":{"do_sample":''' + str(dosample) + \
                            r''',"max_tokens_to_generate":''' + str(maxtok) + \
                            r''',"process_prompt":''' + str(processprompt) + \
                            r''',"repetition_penalty":''' + str(penalty) + \
                            r''',"select_expert":"''' + str(expert) + \
                            r'''","temperature":''' + str(temperature) + \
                            r''',"top_k":''' + str(topk) + \
                            r''',"top_p":''' + str(topp) + \
                            r'''}}'''
        else:
            if streamflag == True:
                datapart = r'''{"items":[{"id":"item1", "value":"{\"conversation_id\":\"sambaverse-conversation-id\",\"messages\":[{\"message_id\":0,\"role\":\"user\",\"content\":\"''' + str(prompt) + \
                            r'''\"}]}"}],"params":{"do_sample":''' + str(dosample) + \
                            r''',"max_tokens_to_generate":''' + str(maxtok) + \
                            r''',"process_prompt":''' + str(processprompt) + \
                            r''',"repetition_penalty":''' + str(penalty) + \
                            r''',"select_expert":"''' + str(expert) + \
                            r'''","temperature":''' + str(temperature) + \
                            r''',"top_k":''' + str(topk) + \
                            r''',"top_p":''' + str(topp) + \
                            r'''}}'''
            else:
                datapart = r'''{"items":[{"id":"item1", "value":"{\"conversation_id\":\"sambaverse-conversation-id\",\"messages\":[{\"message_id\":0,\"role\":\"user\",\"content\":\"''' + str(prompt) + \
                            r'''\"}]}"}],"params":{"do_sample":''' + str(dosample) + \
                            r''',"max_tokens_to_generate":''' + str(maxtok) + \
                            r''',"process_prompt":''' + str(processprompt) + \
                            r''',"repetition_penalty":''' + str(penalty) + \
                            r''',"select_expert":"''' + str(expert) + \
                            r'''","temperature":''' + str(temperature) + \
                            r''',"top_k":''' + str(topk) + \
                            r''',"top_p":''' + str(topp) + \
                            r'''}}'''

    return datapart

def build_curl_cmd(max_words, prompt, expert, dosample, processprompt, key, streamflag):
    data = build_prompt_data_part(max_words, prompt, expert, dosample, processprompt, streamflag)
       
    header = {"key": key, "Content-Type": "application/json"}

    return header, data

def execute_curl_cmd(max_words, prompt, expert, dosample, processprompt, key, url):
    streamflag = True
    if 'stream/' not in url:
        streamflag = False

    headers, data = build_curl_cmd(max_words, prompt, expert, dosample, processprompt, key, streamflag)
    # print(headers, data)

    events = None

    start = time.time()
    end = start

    session = requests.Session()
    try:
        response = session.post(
            url,
            headers=headers,
            data=data.encode(),
            stream=streamflag)
        response.raise_for_status() 
    except requests.exceptions.HTTPError as errh: 
        print("HTTP Error") 
        print(errh.args[0]) 
        detailed = None
        try:
            for idx, line in enumerate(response.iter_lines(decode_unicode=True)):
                detailed = line
        except:
            pass
        return 'None', start, start, time.time(), 'None', f"HTTPError:{str(errh)}+{detailed}", 'None'
    except requests.exceptions.ReadTimeout as errrt: 
        print("Time out") 
        print(errrt.args[0])
        detailed = None
        try:
            for idx, line in enumerate(response.iter_lines(decode_unicode=True)):
                detailed = line
        except:
            pass
        return 'None', start, start, time.time(), 'None', f"Timeout:{str(errrt)}", 'None'
    except requests.exceptions.ConnectionError as conerr: 
        print("Connection error") 
        print(conerr.args[0])
        detailed = None
        try:
            for idx, line in enumerate(response.iter_lines(decode_unicode=True)):
                detailed = line
        except:
            pass
        return 'None', start, start, time.time(), 'None', f"ConnectionError:{str(conerr)}", 'None'
    
    cnt = 0
    lastline = None

    ## for now, just separate stream and completion modes
    if streamflag == True:
        for idx, line in enumerate(response.iter_lines(decode_unicode=True)):
            cnt = idx + 1
            first_event_time_tmp = time.time()
            if idx == 0:
                first_event_time = time.time()
                first_event_time_tmp = first_event_time
                response_data = json.loads(line)
                try:
                    app_first_event_time = -1
                except Exception as e:
                    print(f'Data: {data}\nResponse Data: {response_data}')
                    print("URL: ", url)
                    return 'None', start, first_event_time_tmp, time.time(), 'None', 'None', 'None'

            if "\"is_last_response\":true" in line:
                end = time.time()
                events = line
                print(events)

                return data, start, first_event_time, end, cnt, events, app_first_event_time
            else:
                end = time.time() 
                lastline = line
        return data, start, first_event_time, end, cnt, lastline, app_first_event_time
    else:
        for idx, line in enumerate(response.iter_lines(decode_unicode=True)):
            if "\"id\":\"item1" in line:
                end = time.time()
                events = line
                response_data = json.loads(line)
                app_first_event_time = end
                print(events)

                return data, start, end, end, cnt, events, app_first_event_time
        
        return data, start, end, end, 1, 'None', 'None'

        
def task(task_id, writer_lock, results_list, sleep_time, max_words_list, prompt, expert, dosample, processprompt, key, url, rdu_o=None, golden=None, score=None):
    """Thread task to execute curl commands and log results."""
    max_words = random.choice(max_words_list)

    data, start_time, first_event_time, end_time, msgcnt, events, e2ettft = execute_curl_cmd(max_words, prompt, expert, dosample, processprompt, key, url)

    if 'ConnectionError' in events or 'Timeout' in events or 'HTTPError' in events:
        with writer_lock:
            if rdu_o or golden:
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, events, events, -1, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', 'None', rdu_o, golden, score, 'None'])
            else:
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, events, events, -1, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', 'None'])

        return

    elif events == 'None':
        with writer_lock:
            if rdu_o or golden:
                # golden = clean_string(golden)
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, 'PredictError', 'PredictError', -1, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', 'None', rdu_o, golden, score, 'None'])
            else:
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, 'PredictError', 'PredictError', -1, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', 'None'])

        return
    
    completion, tokenscnt, sidenote, app_total_latency, app_total_tokens_per_sec, app_completion_tokens_after_first_per_sec_first_ten, app_total_tokens_count, app_prompt_tokens_count, app_completion_tokens_after_first_per_sec, app_completion_tokens_per_sec, app_time_to_first_token, app_batch_size_used = parse_curl_response(events, 'stream/' in url)
    
    if completion == 'TimeoutLong':
        with writer_lock:
            if rdu_o or golden:
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, app_prompt_tokens_count, tokenscnt, -1, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', 'None', rdu_o, golden, score, 'None'])
            else:
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, app_prompt_tokens_count, tokenscnt, -1, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', 'None'])

        return

    if tokenscnt >= 0:
        try:    
            tokensec = int(tokenscnt) / float(end_time - start_time)
            afterfirsttokensec = (int(tokenscnt) - int(int(tokenscnt)/msgcnt)) / (end_time - start_time - (first_event_time - start_time))
        except:
            tokensec = 'None'
            afterfirsttokensec = 'None'

        with writer_lock:
            if rdu_o or golden:
                # golden = clean_string(golden)
                golden_value = golden.split('#### ')[-1].strip('\n')
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, app_prompt_tokens_count, tokenscnt, first_event_time-start_time, end_time-start_time, app_time_to_first_token, app_total_latency, tokensec, afterfirsttokensec, app_completion_tokens_per_sec, app_completion_tokens_after_first_per_sec, prompt, app_batch_size_used, completion, rdu_o, golden, score, golden_value.lower() in completion.lower() or completion.lower() in golden_value.lower()])
            else:
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, app_prompt_tokens_count, tokenscnt, first_event_time-start_time, end_time-start_time, app_time_to_first_token, app_total_latency, tokensec, afterfirsttokensec, app_completion_tokens_per_sec, app_completion_tokens_after_first_per_sec, prompt, app_batch_size_used, completion])

    else:
        with writer_lock:
            if rdu_o or golden:
                golden = clean_string(golden)
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, 'EarlyTermination', 'EarlyTermination', first_event_time-start_time, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', events, rdu_o, golden, score, 'None'])
            else:
                results_list.append([datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S"), datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S"), dosample, processprompt, max_words, sleep_time, 'EarlyTermination', 'EarlyTermination', first_event_time-start_time, 'None', 'None', 'None', 'None', 'None', 'None', 'None', prompt, 'None', events])


def perform_post_processing(df):
    """Post-process DataFrame to add calculated columns."""
    # sorted_df = df.sort_values(by=['End Time', 'Start Time'], ascending=[True, True])
    sorted_df = df.sort_values(by=['Start Time', 'End Time'], ascending=[True, True])

    # Temporary conversion to datetime to calculate the differences
    sorted_df['End Time Temp'] = pd.to_datetime(sorted_df['End Time'], format='%H:%M:%S')
    sorted_df['Start Time Temp'] = pd.to_datetime(sorted_df['Start Time'], format='%H:%M:%S')

    # Calculate 'Batch Order'
    # sorted_df['Batch Order'] = sorted_df['End Time Temp'].diff().ne(pd.Timedelta(seconds=0)).cumsum() - 1  # same endtime
    diff = sorted_df['End Time Temp'].diff().fillna(pd.Timedelta(seconds=0))  # Calculate the difference
    sorted_df['Batch Order'] = diff.gt(pd.Timedelta(seconds=0)).cumsum()  # Greater than 1 second difference increments batch number

    sorted_df['Completion Time (sec)'] = (sorted_df['End Time Temp'] - sorted_df['Start Time Temp']).dt.total_seconds().astype(int)

    # Initialize the 'temp' column with the first 'Start Time' value correctly
    sorted_df.at[0, 'temp'] = sorted_df.at[0, 'Start Time']  # Use string value directly

    # Iterate through the DataFrame rows and apply the logic for 'temp' column
    for i in range(1, len(sorted_df)):
        if sorted_df.at[i - 1, 'End Time'] == sorted_df.at[i, 'End Time']:
            # If 'End Times' are the same, replicate the 'temp' value from the previous row
            sorted_df.at[i, 'temp'] = sorted_df.at[i - 1, 'temp']
        elif sorted_df.at[i - 1, 'End Time'] > sorted_df.at[i, 'Start Time']:
            # If the previous 'End Time' is greater than the current 'Start Time', carry forward the previous 'End Time'
            sorted_df.at[i, 'temp'] = sorted_df.at[i - 1, 'End Time']
        else:
            # Otherwise, set 'temp' to the current row's 'Start Time'
            sorted_df.at[i, 'temp'] = sorted_df.at[i, 'Start Time']

    sorted_df['Temp Time Temp'] = pd.to_datetime(sorted_df['temp'], format='%H:%M:%S')
    sorted_df['Queueing Time (sec)'] = (sorted_df['Temp Time Temp'] - sorted_df['Start Time Temp']).dt.total_seconds().astype(int)
    sorted_df['Inference Time (sec)'] = (sorted_df['End Time Temp'] - sorted_df['Temp Time Temp']).dt.total_seconds().astype(int)

    sorted_df.drop(['End Time Temp', 'Start Time Temp', 'Temp Time Temp'], axis=1, inplace=True)

    return sorted_df

def main(max_sleep_time, tasks_num, max_words_list, prompt_l, expert, dosample, processprompt, key, url, goldref_dict=None):
    results_list = []
    writer_lock = threading.Lock()
    
    threads = []
    for i in range(1, tasks_num + 1):
        if max_sleep_time >= 2:
            if i == 1:
                sleep_time = 0 
            else:
                sleep_time = random.randint(1, max_sleep_time)
        else:
            sleep_time = max_sleep_time
        
        if goldref_dict:
            thread = threading.Thread(target=task, args=(i, writer_lock, results_list, sleep_time, max_words_list, prompt_l[i-1], expert, dosample, processprompt, key, url, goldref_dict['rdu_o'][i-1], goldref_dict['golden'][i-1], goldref_dict['score'][i-1]))
        else:
            thread = threading.Thread(target=task, args=(i, writer_lock, results_list, sleep_time, max_words_list, prompt_l[i-1], expert, dosample, processprompt, key, url))
        threads.append(thread)
        time.sleep(sleep_time)
        thread.start()
    for thread in threads:
        thread.join()

    if goldref_dict:
        df = pd.DataFrame(results_list, columns=['Start Time', 'End Time', 'Do sample', 'Process Prompt', 'Max Tokens', 'Request Interval (sec)', 'Input Tokens Cnt', 'Output Tokens Cnt', 'E2E TTFT (sec)', 'E2E Latency (sec)', 'App TTFT (sec)', 'App Latency (sec)', 'E2E tok/s', 'E2E tok/s after first', 'App tok/s', 'App tok/s after first', 'Prompt', 'App batch size used', 'Response', 'ML RDU output', 'GPU golden output', 'ML score', 'Response Match?'])
        df = perform_post_processing(df)

        try:
            df['Queueing Time (sec)'] = (df['E2E Latency (sec)'] - df['App Latency (sec)']).astype(int)
        except:
            pass

        # df = df[['Start Time', 'End Time', 'Do sample', 'Process Prompt', 'Max Tokens', 'Request Interval (sec)', 'App batch size used', 'Batch Order', 'Completion Time (sec)', 'temp', 'Queueing Time (sec)', 'Inference Time (sec)', 'Output Tokens Cnt', 'E2E TTFT (sec)', 'E2E Latency (sec)', 'App TTFT (sec)', 'App Latency (sec)', 'E2E tok/s', 'E2E tok/s after first', 'App tok/s', 'App tok/s after first', 'Prompt', 'Response']]
        df = df[['Start Time', 'End Time', 'Do sample', 'Process Prompt', 'Input Tokens Cnt', 'Output Tokens Cnt', 'Request Interval (sec)', 'App batch size used', 'Batch Order', 'Completion Time (sec)', 'Queueing Time (sec)', 'Max Tokens', 'E2E TTFT (sec)', 'E2E Latency (sec)', 'App TTFT (sec)', 'App Latency (sec)', 'E2E tok/s', 'E2E tok/s after first', 'App tok/s', 'App tok/s after first', 'Prompt', 'ML RDU output', 'GPU golden output', 'ML score', 'Response Match?', 'Response']]

        # df['E2E TTFT (sec)'] = df['E2E TTFT (sec)'] - df['Queueing Time (sec)']
        # df['App TTFT (sec)'] = df['App TTFT (sec)'] - df['Queueing Time (sec)']
    else:
        df = pd.DataFrame(results_list, columns=['Start Time', 'End Time', 'Do sample', 'Process Prompt', 'Max Tokens', 'Request Interval (sec)', 'Input Tokens Cnt', 'Output Tokens Cnt',  'E2E TTFT (sec)', 'E2E Latency (sec)', 'App TTFT (sec)', 'App Latency (sec)', 'E2E tok/s', 'E2E tok/s after first', 'App tok/s', 'App tok/s after first', 'Prompt', 'App batch size used', 'Response'])
        df = perform_post_processing(df)

        try:
            df['Queueing Time (sec)'] = (df['E2E Latency (sec)'] - df['App Latency (sec)']).astype(int)
        except:
            pass

        # df = df[['Start Time', 'End Time', 'Do sample', 'Process Prompt', 'Max Tokens', 'Request Interval (sec)', 'App batch size used', 'Batch Order', 'Completion Time (sec)', 'temp', 'Queueing Time (sec)', 'Inference Time (sec)', 'Output Tokens Cnt', 'E2E TTFT (sec)', 'E2E Latency (sec)', 'App TTFT (sec)', 'App Latency (sec)', 'E2E tok/s', 'E2E tok/s after first', 'App tok/s', 'App tok/s after first', 'Prompt', 'Response']]
        df = df[['Start Time', 'End Time', 'Do sample', 'Process Prompt', 'Input Tokens Cnt', 'Output Tokens Cnt', 'Request Interval (sec)', 'App batch size used', 'Batch Order', 'Completion Time (sec)', 'Queueing Time (sec)', 'Max Tokens', 'E2E TTFT (sec)', 'E2E Latency (sec)', 'App TTFT (sec)', 'App Latency (sec)', 'E2E tok/s', 'E2E tok/s after first', 'App tok/s', 'App tok/s after first', 'Prompt', 'Response']]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a series of asynchronous tasks with variable parameters.")
    parser.add_argument('--max_sleep_time', type=float, default=0, help='Maximum sleep time between tasks in seconds')
    parser.add_argument('--max_words_list', type=int, nargs='+', default=[512], help='List of possible max words for tasks')
    parser.add_argument("-k", "--key", required=True, help="API key")
    parser.add_argument("-u", "--url", required=True, help="URL")
    parser.add_argument("-e", "--expert", required=True, help="Expert to run")
    parser.add_argument("-i", "--instance", required=True, help="Number of instances the endpoint is running on")
    parser.add_argument("-a", "--app", required=False, default="noNeed", help="App of Expert")
    parser.add_argument("-p", "--prompt_file", required=False, default="../../data/json/mixpanel_fastcoe_unique_prompts.json", help="JSON File with list of user prompts")
    parser.add_argument("-m", "--mode", required=False, default="benchmark", help='user select mode from [benchmark|all|gold|negative|gq]')
    parser.add_argument("-r", "--random", required=False, default="Y", help='Y: random select; N: select sequencially')
    parser.add_argument("-s", "--stream", required=False, default="Y", help='Y: streaming mode; N: prompt(completion) mode')
    parser.add_argument("-tn", "--tasknum", type=int, nargs='+', default=[52], help='List of number concurrent tasks')

    args = parser.parse_args()
    expert = args.expert
    app = args.app

    if app != "noNeed":
        expert = app + '+' + expert
    
    prompt_generator = Prompt(args.prompt_file)

    do_sample_list = ['false', 'true']
    process_prompt_list = ['false', 'true']
    
    model_dict = read_master_menu("../../data/csv/samba_turbo_master_menu.csv")
    
    if expert not in model_dict:
        print(f"Expert '{expert}' not found in model_dict {model_dict}.")
        

    if args.mode == 'benchmark' or args.mode == 'gold' or args.mode == 'gq' or args.mode == 'switching':
        do_sample_list = ['false']
        process_prompt_list = ['false']
        
    if args.mode == 'errinjection':
        process_prompt_list = ['true']

    res = []
        
    prompt_l = []

    if args.mode == 'gold':
        goldref_dict = extract_triviaqa_prompts(args.prompt_file)
        prompt_l = goldref_dict['prompt']
    elif args.mode == 'benchmark':
        # Load from prompt files
        if 'code' in expert.lower():
            with open('../../data/yaml/template_deepseek_coder.yaml', 'r') as file:
                yaml_data = yaml.safe_load(file)
        else:
            with open('../../data/yaml/template.yaml', 'r') as file:
                yaml_data = yaml.safe_load(file)
        prompt_template = yaml_data['template']
        
        for prompt_index in range(model_dict[expert]["Batch Sizes"][-1] + 1):
            prompt_l.append(prompt_template)
    else:
        for prompt_index in range(model_dict[expert]["Batch Sizes"][-1] + 200):
            if args.random == 'Y':
                prompt = prompt_generator.get_random_user_prompt()
            else:
                try:
                    prompt = prompt_generator.get_random_user_prompt(prompt_index)
                except:
                    prompt = prompt_generator.get_random_user_prompt(prompt_index % 100)
            prompt_l.append(prompt)

    for dosample in do_sample_list:
        for processprompt in process_prompt_list:
            new_prompt_l = []
            if processprompt == 'false':
                if args.mode == 'gold':
                    for i in range(len(prompt_l)):
                        scrub = prompt_generator.scrub_prompt(prompt_l[i])
                        new_prompt_l.append(prompt_generator.create_prompt(scrub, model_dict[expert]["Prompt Template"]))
                elif args.mode == 'negative':
                    new_prompt_l = prompt_l 
                elif args.mode == 'benchmark':
                    for i in range(len(prompt_l)):
                        scrub = prompt_generator.scrub_prompt(prompt_l[i])  
                        new_prompt_l.append(scrub)                   
                else:
                    for i in range(len(prompt_l)):
                        scrub = prompt_generator.scrub_prompt(prompt_l[i])
                        new_prompt_l.append(prompt_generator.create_prompt(scrub, model_dict[expert]["Prompt Template"]))
            else:
                if args.mode == 'negative':
                    for i in range(len(prompt_l)):
                        scrub = prompt_generator.scrub_prompt(prompt_l[i])
                        new_prompt_l.append(prompt_generator.create_prompt(scrub, model_dict[expert]["Prompt Template"]))
                elif args.mode == 'errinjection':
                    for i in range(len(prompt_l)):
                        scrub = prompt_generator.scrub_prompt(prompt_l[i])
                        new_prompt_l.append(prompt_generator.create_prompt(scrub, model_dict[expert]["Prompt Template"]))
                else:
                    for i in range(len(prompt_l)):
                        scrub = prompt_generator.scrub_prompt(prompt_l[i])
                        new_prompt_l.append(scrub)

            for tasks_num in args.tasknum:
                if args.mode == 'gold':
                    tasks_num = len(new_prompt_l)
                    expert_use = expert.split('+')[-1]
                    if args.stream == 'Y':
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url, goldref_dict)
                    else:
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url.replace('stream/',''), goldref_dict)
                    res.append(df)
                elif args.mode == 'benchmark':
                    if args.stream == 'Y':
                        graph_3_list = []
                        if model_dict[expert]["3-graph"][-1] > 4096:
                            graph_3_list = [200, 1000, 7000]
                        elif model_dict[expert]["3-graph"][-1] > 1024:
                            graph_3_list = [200, 1000]
                        else:
                            graph_3_list = [200]

                        for input_tokens_num in graph_3_list:
                            tokenized_prompt = prompt_generator.get_tokenizer(model_dict[expert]["model_arch"], model_dict[expert]["Prompt Template"], new_prompt_l[0], input_tokens_num)        

                            benchmark_prompt_l = [tokenized_prompt] * len(new_prompt_l)

                            for tasks_num in model_dict[expert]["Batch Sizes"]:
                                tasks_num += 1
                                expert_use = expert.split('+')[-1]
                                df = main(args.max_sleep_time, tasks_num, args.max_words_list, benchmark_prompt_l, expert_use, dosample, processprompt, args.key, args.url)
                                tasks_num -= 1
                                filename = f'{expert_use}_{args.mode}_stream{args.stream}_qaTemplate_random{args.random}_delay{args.max_sleep_time}_numInTokens{input_tokens_num}_numOutTokens{args.max_words_list[0]}_requests{tasks_num}_instance{args.instance}'
                                df.to_csv(f'./reports/{filename}.csv', index=False, escapechar='/')
                    
                    else:
                        expert_use = expert.split('+')[-1]
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url.replace('stream/',''))

                    exit(1)
                elif args.mode == 'cebenchmark':
                    expert_use = expert.split('+')[-1]
                    if args.stream == 'Y':
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url)
                    else:
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url.replace('stream/',''))
                    res.append(df)
                elif args.mode == 'switching':
                    expert_use = expert.split('+')[-1]
                    if args.stream == 'Y':
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url)
                    else:
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url.replace('stream/',''))
                    res.append(df)
                else:
                    expert_use = expert.split('+')[-1]
                    if args.stream == 'Y':
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url)
                    else:
                        df = main(args.max_sleep_time, tasks_num, args.max_words_list, new_prompt_l, expert_use, dosample, processprompt, args.key, args.url.replace('stream/',''))
                    res.append(df)
        
        df_res = pd.concat(res)
        expert_use = expert.split('+')[-1]
        filename = f'{expert_use}_{args.mode}_stream{args.stream}_{args.prompt_file.split("_")[1]}_random{args.random}_delay{args.max_sleep_time}_maxwords{args.max_words_list[0]}_instance{args.instance}'
        df_res.to_csv(f'./reports/{filename}.csv', index=False, escapechar='/')