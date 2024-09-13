import subprocess
import itertools
import yaml
import random
import time
import os


class GeneralModelTestObject():
    def __init__(self, project, chiparch):
        self.project = project 
        self.arch = chiparch
        self.forbiddenSweep = {'do_eval', 'evaluation_strategy', 'lr_schedule', 
                               'save_optimizer_state', 'model_arch_type',
                               'use_token_type_ids', 'truncate_pattern',
                               'scheduler_type', 'normalize', 'use_lm_decoding',
                               'use_number_transcriber', 'prediction_handler',
                               'max_seq_length'}
                            #    'debug_mode', 'dump_inputs', 'fix_rank_rdu_mapping'}

    def model_config_generation_prepare(self, configFile, sampleNum=1):

        resConfigDict = {}
        
        with open(configFile, 'r') as f:
            configDict = yaml.safe_load(f)

            try:
                configDict.items()
            except:
                return resConfigDict
            
            for k, v in configDict.items():
                try:
                    subkeys = list(configDict[k].keys())
                except:
                    # TODO: STOP_SEQUENCE = None
                    continue
                if 'values' in subkeys:
                    if sampleNum > 1:
                        resConfigDict[k] = configDict[k]['values']
                    else:
                        resConfigDict[k] = [configDict[k]['values'][0]]

                    if k == 'evaluation_strategy':
                        resConfigDict[k] = ['steps']
                elif len(subkeys) == 1:
                    ##TODO: how to set correct range for a hyperparameter in model info list? 
                    if 'ge' in subkeys:
                        try:
                            if configDict[k]['ge'] == '0' and 'step' not in k:
                                resConfigDict[k] = [random.uniform(float(configDict[k]['ge']), 1.0) for _ in range(sampleNum)]
                            else:
                                resConfigDict[k] = [random.randrange(int(configDict[k]['ge']), int(configDict[k]['ge'])+2) for _ in range(sampleNum)]
                        except:
                            resConfigDict[k] = [random.randrange(float(configDict[k]['ge']), float(configDict[k]['ge'])+100.0) for _ in range(sampleNum)] 
                    elif 'gt' in subkeys:
                        try:
                            resConfigDict[k] = [random.randrange(int(configDict[k]['gt'])+0, int(configDict[k]['gt'])+100) for _ in range(sampleNum)]
                        except:
                            resConfigDict[k] = [random.randrange(float(configDict[k]['gt'])+0.01, float(configDict[k]['gt'])+100.0) for _ in range(sampleNum)]
                    if 'le' in subkeys:
                        try:
                            resConfigDict[k] = [random.randrange(0, int(configDict[k]['le'])) for _ in range(sampleNum)]
                        except:
                            resConfigDict[k] = [random.randrange(0, float(configDict[k]['le'])) for _ in range(sampleNum)]
                    elif 'lt' in subkeys:
                        try:
                            resConfigDict[k] = [random.randrange(0, int(configDict[k]['lt'])) for _ in range(sampleNum)]
                        except:
                            resConfigDict[k] = [random.randrange(0, float(configDict[k]['lt'])) for _ in range(sampleNum)]     
                elif len(subkeys) == 2:
                    lowerbound = float(configDict[k][subkeys[0]])+ 1e-9
                    upperbound = float(configDict[k][subkeys[1]])

                    if upperbound != '1':
                        resConfigDict[k] = [int(random.uniform(lowerbound, upperbound)) for _ in range(sampleNum)]
                    else:
                        resConfigDict[k] = [random.uniform(lowerbound, upperbound) for _ in range(sampleNum)]

        return resConfigDict
    
    def model_config_generation(self, modelName, configFileList, sampleNum=1):
        training_hf_l = []
        inference_hf_l = []


        if not configFileList or len(configFileList) == 0:
            return

        for configFile in configFileList:
            resConfigDict = self.model_config_generation_prepare(configFile, sampleNum=1)

            key_l = list(resConfigDict.keys())
            hf_l = [v for k, v in resConfigDict.items()]
            hfcombo_l = list(itertools.product(*hf_l))

            if 'train' in configFile:
                testType = 'training'
            elif 'predict' in configFile:
                testType = 'inference'

            for i, hfitem in enumerate(hfcombo_l):
                modelName = modelName.replace(' ', '_')
                filename = f"../../artifacts/hf/{modelName}_{testType}_testcase{i+1}.yaml"
                # for i, v in enumerate(key_l):
                #     if i < len(key_l) - 1:
                #         filename += f"{v.replace('_','')}{hfitem[i]}_"
                #     else:
                #         filename += f"{v}{hfitem[i]}.yaml"

                if testType == 'training':
                    training_hf_l.append(filename)
                elif testType == 'inference':
                    inference_hf_l.append(filename)

                f = open(f"{filename}", "w")
                for i, v in enumerate(key_l):
                    f.write(f"{v}: {hfitem[i]}\n")
                f.close()

        return training_hf_l, inference_hf_l

    def model_config_generation_new(self, modelName, configFileList, MODE='NORMAL'):
        training_hf_l = []
        inference_hf_l = []

        modelName = modelName.replace(' ', '_')

        if not configFileList or len(configFileList) == 0:
            return training_hf_l, inference_hf_l

        for configFile in configFileList:
            resConfigDict = {}

            try:
                f=open(configFile, 'r')
            except:
                continue

            with open(configFile, 'r') as f:
                configDict = yaml.safe_load(f)
                print(configDict)

                try:
                    configDict.items()
                except:
                    return training_hf_l, inference_hf_l
                
                for k, v in configDict.items():
                    
                    try:
                        subkeys = list(configDict[k].keys())
                    except:
                        # TODO: STOP_SEQUENCE = None
                        continue
                    if 'values' in subkeys and k not in self.forbiddenSweep:
                        if k == 'vocab_size' and '300k' not in modelName and 'GPT' in modelName:
                            resConfigDict[k] = ['50260']
                        elif k == 'max_seq_length' and '8k' in modelName:
                            resConfigDict[k] = ['8192']
                        elif k == 'max_seq_length' and '2k' in modelName:
                            resConfigDict[k] = ['2048']
                        elif k == 'max_seq_length' and ('2k' not in modelName and '8k' not in modelName and 'Llama' not in modelName and 'Hubert' not in modelName and '1.5B' not in modelName and 'GPT_13B_Base_Model' not in modelName):
                            resConfigDict[k] = ['2048']
                        elif k == 'skip_checkpoint':
                            resConfigDict[k] = ['true']
                        else:
                            if configDict[k]['values'] or len(configDict[k]['values']) > 0:
                                resConfigDict[k] = configDict[k]['values']

            key_l = list(resConfigDict.keys())
            hf_l = [v for k, v in resConfigDict.items()]
            hfcombo_l = list(itertools.product(*hf_l))

            if 'train' in configFile:
                testType = 'training'
            elif 'predict' in configFile:
                testType = 'inference'

            for i, hfitem in enumerate(hfcombo_l):
                filename = f"../../artifacts/hf/{modelName}_{testType}"

                if not key_l or len(key_l) == 0:
                    filename += ".yaml"
                else:
                    for i, v in enumerate(key_l):
                        if i < len(key_l) - 1:
                            filename += f"_{v.replace('_','')}{hfitem[i]}"
                        else:
                            filename += f"_{v.replace('_','')}{hfitem[i]}.yaml"

                if testType == 'training':
                    training_hf_l.append(filename)
                elif testType == 'inference':
                    inference_hf_l.append(filename)

                f = open(f"{filename}", "w")

                if not key_l or len(key_l) == 0:
                    if 'train' in filename:
                        if MODE == 'ND':
                             f.write(f"num_iterations: 10\n")
                             f.write(f"logging_steps: 1\n")
                             f.write(f"do_eval: false\n")
                             f.write(f"skip_checkpoint: true\n")
                            #  f.write('''evaluation_strategy: "no"\n''')
                        elif 'CKPT' in MODE:
                             f.write(f"num_iterations: 10\n")
                             f.write(f"logging_steps: 1\n")
                             f.write(f"do_eval: false\n")
                             if '1.5b' not in modelName.lower():
                                f.write(f"skip_checkpoint: false\n")
                             f.write(f"save_optimizer_state: true\n")
                             f.write(f"save_steps: 10\n")
                             f.write(f"evaluation_strategy: no\n")
                        else:
                            # the following models are very slow 300 iteration/sec, so just run 10 steps
                            if 'llama-2-13b' in modelName.lower() or \
                               'llama-2-7b-16k' in modelName.lower() or \
                               'llama-2-7b-chat-16k' in modelName.lower():
                                f.write(f"num_iterations: 10\n")
                            else:
                                f.write(f"num_iterations: 10\n")

                    f.close()

                else:
                    for i, v in enumerate(key_l):
                        if 'CKPT' not in MODE:
                            f.write(f"{v}: {hfitem[i]}\n")
                        if 'train' in filename and ('gpt' in filename.lower() or 'llama' in filename.lower()):
                            if MODE == 'ND':
                                f.write(f"num_iterations: 10\n")
                                f.write(f"logging_steps: 1\n")
                                f.write(f"do_eval: false\n")
                                f.write(f"skip_checkpoint: true\n")
                                #  f.write('''evaluation_strategy: "no"\n''')
                            elif 'CKPT' in MODE:
                                f.write(f"num_iterations: 10\n")
                                f.write(f"logging_steps: 1\n")
                                f.write(f"do_eval: false\n")
                                if '1.5b' not in modelName.lower():
                                    f.write(f"skip_checkpoint: false\n")
                                f.write(f"save_optimizer_state: true\n")
                                f.write(f"save_steps: 10\n")
                                f.write(f"evaluation_strategy: no\n")
                            else:
                                # the following models are very slow 300 iteration/sec, so just run 10 steps
                                if 'llama-2-13b' in modelName.lower() or \
                                'llama-2-7b-16k' in modelName.lower() or \
                                'llama-2-7b-chat-16k' in modelName.lower():
                                    f.write(f"num_iterations: 10\n")
                                else:
                                    f.write(f"num_iterations: 10\n")
                                f.write(f"skip_checkpoint: true\n")
                    f.close()

        return training_hf_l, inference_hf_l

    def model_training(self, fileHandler, training_hf_l, modelName, dataName, RDU=1, SWEEP='N'):
        
        res = 'None'

        if len(training_hf_l) == 0 or SWEEP=='N':
            if 'GPT' in modelName and ('8k' in modelName.lower() or '8192' in modelName.lower()) and '8k' not in dataName.lower():
                print(self.project, modelName, 'Training', 'DataMismatch', dataName, SWEEP, 'None', command)
                fileHandler.write(f"{self.project},{modelName},{modelName.replace(' ', '_')}_training_sweep_{SWEEP}_{dataName},Training,'DataMismatch',{dataName},{SWEEP},None,{command.replace(',',' ')}\n")
                return fileHandler
            
            if 'GPT' in modelName and ('2k' in modelName.lower() or '2048' in modelName.lower()) and '8k' in dataName.lower():
                print(self.project, modelName, 'Training', 'DataMismatch', dataName, SWEEP, 'None', command)
                fileHandler.write(f"{self.project},{modelName},{modelName.replace(' ', '_')}_training_sweep_{SWEEP}_{dataName},Training,'DataMismatch',{dataName},{SWEEP},None,{command.replace(',',' ')}\n")
                return fileHandler

            command = "snapi job create " + \
                        f"-p {self.project} " + \
                        f"-j {modelName.replace(' ', '_')}_training_sweep_{SWEEP}_{dataName}_rdu{RDU} " + \
                        f"-t train " + \
                        f"-m '{modelName}' " + \
                        f"-d {dataName} " + \
                        f"-r {RDU} " + \
                        f"-a {self.arch} " + \
                        ''' -hp '{"num_iterations":"10", "skip_checkpoint": "true", "save_optimizer_state": "false"}' '''

            res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = res.communicate()
            
            if error:
                print("Error:", error.decode())

            for line in output.decode().splitlines():
                if 'Successfully created' in line:
                    res = 'Succeed'
                    break
                elif 'Duplicate job' in line:
                    res = 'Duplicated'
                    break
                else:
                    res = 'Failed'
                    break
            print(self.project, modelName, 'Training', res, dataName, SWEEP, 'None', command)
            fileHandler.write(f"{self.project},{modelName},{modelName.replace(' ', '_')}_training_sweep_{SWEEP}_{dataName}_rdu{RDU},Training,{res},{dataName},{SWEEP},None,{command.replace(',',' ')}\n")
            
            return fileHandler

        for hf in training_hf_l:
            # TODO: GPT 13B dataset confusion from Modelbox, just for now. Delete it.
            if 'GPT' in hf and ('8k' in hf.lower() or '8192' in hf.lower()) and '8k' not in dataName.lower():
                continue
            if 'GPT' in hf and ('2k' in hf.lower() or '2048' in hf.lower()) and '8k' in dataName.lower():
                continue

            jobname = hf.split('.yaml')[0].split('../../artifacts/hf/')[-1] + f"_{dataName.replace(' ','').replace('_','')}_{RDU}rdu_sweep_{SWEEP}"

            if SWEEP != 'N':
                command = "snapi job create " + \
                        f"-p {self.project} " + \
                        f"-j {jobname} " + \
                        f"-t train " + \
                        f"-m '{modelName}' " + \
                        f"-d {dataName} " + \
                        f"-hf {hf} " + \
                        f"-r {RDU} " + \
                        f"-a {self.arch}"
            else:
                command = "snapi job create " + \
                        f"-p {self.project} " + \
                        f"-j {jobname} " + \
                        f"-t train " + \
                        f"-m '{modelName}' " + \
                        f"-d {dataName} " + \
                        f"-r {RDU} " + \
                        f"-a {self.arch}"
                                    
            res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = res.communicate()
            
            if error:
                print("Error:", error.decode())

            for line in output.decode().splitlines():
                if 'Successfully created' in line:
                    res = 'Succeed'
                    break
                elif 'Duplicate job' in line:
                    res = 'Duplicated'
                    break
                else:
                    res = 'Failed'
                    break

            print(self.project, modelName, 'Training', res, dataName, SWEEP, hf, command)
            if SWEEP == 'N':
                fileHandler.write(f"{self.project},{modelName},{jobname},Training,{res},{dataName},{SWEEP},None,{command}\n")
            else:
                fileHandler.write(f"{self.project},{modelName},{jobname},Training,{res},{dataName},{SWEEP},{hf},{command}\n")

        return fileHandler

    def model_inference(self, fileHandler, inference_hf_l, modelName, dataName, SWEEP='N'):

        res = 'None'

        if len(inference_hf_l) == 0 or SWEEP == 'N':
            command = "snapi job create " + \
                        f"-p {self.project} " + \
                        f"-j {modelName.replace(' ', '_')}_inference_sweep_{SWEEP}_{dataName} " + \
                        f"-t batch_predict " + \
                        f"-m '{modelName}' " + \
                        f"-d {dataName} " + \
                        f"-a {self.arch}"
            print("=======command: ", command)
            res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = res.communicate()
            
            if error:
                print("Error:", error.decode())

            for line in output.decode().splitlines():
                if 'Successfully created' in line:
                    res = 'Succeed'
                    break
                elif 'Duplicate job' in line:
                    res = 'Duplicated'
                    break
                else:
                    res = 'Failed'
                    break
            print("====job info: ", self.project, modelName, 'Inference', res, dataName, SWEEP, 'None', command)
            fileHandler.write(f"{self.project},{modelName},{modelName.replace(' ', '_')}_inference_sweep_{SWEEP}_{dataName},Inference,{res},{dataName},{SWEEP},None,{command}\n")
            return fileHandler
        
        for hf in inference_hf_l:

            jobname = hf.split('.yaml')[0].split('../../artifacts/hf/')[-1] + f"_{dataName.replace(' ','').replace('_','')}_sweep_{SWEEP}"

            if os.stat(hf).st_size == 0:
                command = "snapi job create " + \
                        f"-p {self.project} " + \
                        f"-j {jobname} " + \
                        f"-t batch_predict " + \
                        f"-m '{modelName}' " + \
                        f"-d {dataName} " + \
                        f"-a {self.arch}"
            
                res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                output, error = res.communicate()

            else:
                if SWEEP != 'N':
                    command = "snapi job create " + \
                            f"-p {self.project} " + \
                            f"-j {jobname} " + \
                            f"-t batch_predict " + \
                            f"-m '{modelName}' " + \
                            f"-d {dataName} " + \
                            f"-hf {hf} " + \
                            f"-a {self.arch}"
                else:
                    command = "snapi job create " + \
                            f"-p {self.project} " + \
                            f"-j {jobname} " + \
                            f"-t batch_predict " + \
                            f"-m '{modelName}' " + \
                            f"-d {dataName} " + \
                            f"-a {self.arch}"
                
                res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                output, error = res.communicate()
            
            if error:
                print("Error:", error.decode())

            for line in output.decode().splitlines():
                if 'Successfully created' in line:
                    res = 'Succeed'
                    break
                elif 'Duplicate job' in line:
                    res = 'Duplicated'
                    break
                else:
                    res = 'Failed'
                    break
            print(self.project, modelName, 'Inference', res, dataName, SWEEP, hf, command)
            if SWEEP == 'N':
                fileHandler.write(f"{self.project},{modelName},{jobname},Inference,{res},{dataName},{SWEEP},None,{command}\n")
            else:
                fileHandler.write(f"{self.project},{modelName},{jobname},Inference,{res},{dataName},{SWEEP},{hf},{command}\n")

        return fileHandler
    
    def model_endpoint(self, modelName):
        ins = 1
        jobname = f"{modelName.replace(' ','-').replace('_','-')}-{ins}ins-endpoint-{self.arch.lower()}".lower()
        # #TODO: DELETE
        # modelName = modelName.split('_')[0]

        try:
            command = "snapi endpoint create " + \
                        f"-p {self.project} " + \
                        f"-n {jobname} " + \
                        f"-m '{modelName}' " + \
                        f"-a {self.arch} " + \
                        f"-i {ins} "
            
            print(command)
            
            res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = res.communicate()
        except:
            print("no model !!!!!!!!")
            return False, None, None

        if error:
            print("Error:", error.decode())
        else:
            for line in output.decode().splitlines():
                if 'Successfully' in str(line):
                    time.sleep(5)
                else:
                    return False, None, None

        # check info of endpoint and extract info
        command = f"snapi endpoint info -p {self.project} -e {jobname}"

        url = None
        apikey = None
        status = None

        counter = 1

        res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = res.communicate()

        if error:
            print("Error:", error.decode())
        else:
            for line in output.decode().splitlines():
                if 'API Key' in str(line) and 'Keys' not in str(line):
                    tmp = str(line).split(': ')
                    apikey = tmp[-1]
                elif 'Status' in str(line):
                    tmp = str(line).split(': ')
                    status = tmp[-1]
                elif 'URL' in str(line):
                    tmp = str(line).split(': ')
                    url = tmp[-1]

        if status == 'Live':
            return True, apikey, url
        elif status == 'Failed':
            return False, apikey, url
        else:
            while status != 'Live' and status != 'Failed' and counter <= 90:
                res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                output, error = res.communicate()
                if error:
                    print("Error:", error.decode())
                else:
                    for line in output.decode().splitlines():
                        if 'Status' in str(line):
                            tmp = str(line).split(': ')
                            status = tmp[-1]
                            break
                    time.sleep(10)
                    counter += 1

            if counter == 91:
                print(f"{jobname} is NOT alive.")
                return False, apikey, url
            else:
                print(f"{jobname} is alive.")
                return True, apikey, url
            
    def model_ndscreening(self, fileHandler, training_hf_l, modelName, dataName, RDU=1, NUMRDU=8):
        
        res = 'None'

        if len(training_hf_l) == 0:
            print(f"There is no training hf for model {modelName}.")
            return None, None
                
        for hf in training_hf_l:
            # TODO: GPT 13B dataset confusion from Modelbox, just for now. Delete it.
            if 'GPT' in hf and ('8k' in hf.lower() or '8192' in hf.lower()) and '8k' not in dataName.lower():
                continue
            if 'GPT' in hf and ('2k' in hf.lower() or '2048' in hf.lower()) and '8k' in dataName.lower():
                continue
            
            numtests = int(NUMRDU/RDU)

            jobname_list = []

            if ('8k' in hf or '8192' in hf) and 'GPT' in hf:
                dataName = 'GPT_13B_8k_SS_Toy_Training_Dataset'

            for i in range(numtests):
                jobname = hf.split('.yaml')[0].split('../../artifacts/hf/')[-1] + f"_{dataName.replace(' ','').replace('_','')}_{RDU}rdu_test_{i}"
                
                command = "snapi job create " + \
                            f"-p {self.project} " + \
                            f"-j {jobname} " + \
                            f"-t train " + \
                            f"-m '{modelName}' " + \
                            f"-d {dataName} " + \
                            f"-hf {hf} " + \
                            f"-r {RDU} " + \
                            f"-a {self.arch}"
                
                res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                output, error = res.communicate()

                if error:
                    print("Error:", error.decode())

                for line in output.decode().splitlines():
                    if 'Successfully created' in line:
                        res = 'Succeed'
                        jobname_list.append(jobname)
                        break
                    elif 'Duplicate job' in line:
                        res = 'Duplicated'
                        break
                    else:
                        res = 'Failed'
                        break

                print(self.project, modelName, 'Training', res, dataName, hf, command)

                fileHandler.write(f"{self.project},{modelName},{jobname},Training,{res},{dataName},{hf},{command}\n")

            return fileHandler, jobname_list
        
    def model_loadckpt(self, hf, modelName, dataName, RDU=1):
        
        if ('8k' in hf or '8192' in hf) and 'GPT' in hf:
            dataName = 'GPT_13B_8k_SS_Toy_Training_Dataset'

        # launch training job
        jobname = hf.split('.yaml')[0].split('../../artifacts/hf/')[-1] + f"_{dataName.replace(' ','').replace('_','')}_{RDU}rdu"
        
        command = "snapi job create " + \
                    f"-p {self.project} " + \
                    f"-j {jobname} " + \
                    f"-t train " + \
                    f"-m '{modelName}' " + \
                    f"-d {dataName} " + \
                    f"-hf {hf} " + \
                    f"-r {RDU} " + \
                    f"-a {self.arch}"

        print("!!!!!!!!!!!!!!!", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print("Error:", error.decode())
        else:
            for line in output.decode().splitlines():
                print(str(line))
                if 'Successfully created' in str(line):
                    res = 'Succeed'
                    break
                elif 'Duplicate job' in str(line):
                    res = 'Duplicated'
                    break
                else:
                    res = 'Failed'
                    return 'InitTrainingFailed'   

        print(self.project, modelName, 'Training', res, dataName, hf, command)

        # check training job until it's done
        flag = 1
        ckptflag = 1

        if res == 'Succeed':
            while flag:
                command = f"snapi job info -p {self.project} -j {jobname}" 

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                output, error = process.communicate()

                if error:
                    print("Error:", error.decode())
                else:
                    for line in output.decode().splitlines():
                        if 'Status' in str(line):
                            tmp = [i for i in str(line).split(' ') if i]
                            
                            if 'STOPPED' in tmp[2]:
                                ckptflag = 0
                                flag = 0
                                break
                            elif 'EXIT_WITH_0' not in tmp[2] and 'FAILED' not in tmp[2]:
                                print(f"job current status: {tmp[2]}, please wait 2 more minutes before next check.")
                                time.sleep(120)
                                break
                            elif 'FAILED' in tmp[2]:
                                ckptflag = 0
                                flag = 0
                            elif 'EXIT_WITH_0' in tmp[2]:
                                ckptflag = 1
                                flag = 0
        
        # check checkpoint info
        ckptid = None

        if ckptflag == 1:
            command = f"snapi checkpoint list -p {self.project} -j {jobname}"
            
            ckptid = None

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error:
                print("Error:", error.decode())
            else:
                for line in output.decode().splitlines():
                    tmp = [i for i in str(line).split(' ') if i]
                    ckptid = tmp[0]
            print(ckptid)

        # save ckpt to modelhub
        if ckptid:
            newmodel = f"test-{ckptid}"
            command = f"snapi model add -m {ckptid} -n {newmodel} -p {self.project} -j {jobname} -t pretrained"

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error:
                print("Error:", error.decode())
            else:
                pass

            if '70' in modelName.lower():
                time.sleep(3000)
            else:
                time.sleep(1000)

            # delete old job
            command = f"snapi job delete -p {self.project} -j {jobname}"

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error:
                print("Error:", error.decode())
            else:
                pass

            time.sleep(120)
        else:
            return 'NoCKPT'

        # start a new job with -l
        command = "snapi job create " + \
                    f"-p {self.project} " + \
                    f"-j {jobname} " + \
                    f"-t train " + \
                    f"-m '{newmodel}' " + \
                    f"-d {dataName} " + \
                    f"-hf {hf} " + \
                    f"-r {RDU} " + \
                    f"-l " + \
                    f"-a {self.arch}"
        
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print("Error:", error.decode())
        else:
            for line in output.decode().splitlines(): 
                if 'Successfully created' in str(line):
                    res = 'Succeed'
                    break
                elif 'Duplicate job' in str(line):
                    res = 'Duplicated'
                    break
                else:
                    res = 'Failed'
                    break

        # check training job until it's done
        flag = 1
        ckptflag = 1

        runres = None

        if res == 'Succeed':
            while flag:
                command = f"snapi job info -p {self.project} -j {jobname}"

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                output, error = process.communicate()

                if error:
                    print("Error:", error.decode())
                else:
                    for line in output.decode().splitlines():     
                        if 'Status' in str(line):
                            tmp = [i for i in str(line).split(' ') if i]
                            
                            if 'STOPPED' in tmp[2]:
                                ckptflag = 0
                                flag = 0
                                runres = 'STOPPED'
                                break
                            elif 'EXIT_WITH_0' not in tmp[2] and 'FAILED' not in tmp[2]:
                                print(f"job current status: {tmp[2]}, please wait 2 more minutes before next check.")
                                time.sleep(120)
                                break
                            elif 'FAILED' in tmp[2]:
                                runres = 'Fail'
                                flag = 0
                            elif 'EXIT_WITH_0' in tmp[2]:
                                runres = 'Success'
                                flag = 0
        
        if runres == 'Success':
            command = f"snapi job delete -p {self.project} -j {jobname}"
            
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error:
                print("Error:", error.decode())
            else:
                pass

        command = f"snapi model remove -m {newmodel}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print("Error:", error.decode())
        else:
            pass

        return runres
    