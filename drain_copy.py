''' Written by Pinjia He, Modified by Optic Team '''
import hashlib
import os.path
import re
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

import logging

from pandas.errors import ParserError

logger = logging.getLogger('root')
fmt = "[ %(asctime)s %(filename)s:%(lineno)s - %(funcName)s() ] %(levelname)s %(message)s"
logging.basicConfig(filename="ml_pipeline.log", filemode='w', format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.DEBUG)


class Logcluster:
    """
    Drain Log Cluster
    """

    def __init__(self, log_template='', log_idl=None):
        self.log_template = log_template
        if log_idl is None:
            log_idl = []
        self.log_idl = log_idl


class Node:
    """
    Drain Node
    """

    def __init__(self, child_d=None, depth=0, digit_or_token=None):
        if child_d is None:
            child_d = dict()
        self.child_d = child_d
        self.depth = depth
        self.digit_or_token = digit_or_token


class Parser:
    """
    Drain Parser
    """

    def __init__(self, log_format, indir='./', depth=4, st=0.4,
                 max_child=100, rex=None, keep_para=True):
        """

        :param log_format: str, Format of the log file template
        :param indir: str, the input path stores the input log file name
        :param outdir: str, the output path stores the file containing structured Logs
        :param depth: int, depth of all leaf nodes
        :param st: float, similarity threshold
        :param max_child: int, max number of children of an internal node
        :param rex: list, regular expressions used in preprocessing (step1)
        """
        curr_dir = os.path.dirname(os.path.dirname(__file__))
        windows_path_output = os.path.join(curr_dir,
                                           'Parsed_Log_Files',
                                           'Parsed_Drain')

        global_path_output = Path(windows_path_output)

        outdir = global_path_output
        print(outdir)
        logger.info(outdir)

        if rex is None:
            rex = []
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.max_child = max_child
        self.log_name = None
        self.save_path = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para

    def has_numbers(self, line):
        return any(char.isdigit() for char in line)

    def tree_search(self, rn, seq):
        ret_log_clust = None

        seq_len = len(seq)
        if seq_len not in rn.child_d:
            return ret_log_clust

        parentn = rn.child_d[seq_len]

        current_depth = 1
        for token in seq:
            if current_depth >= self.depth or current_depth > seq_len:
                break

            if token in parentn.child_d:
                parentn = parentn.child_d[token]
            elif '<*>' in parentn.child_d:
                parentn = parentn.child_d['<*>']
            else:
                return ret_log_clust
            current_depth += 1

        log_clust_l = parentn.child_d

        ret_log_clust = self.fast_match(log_clust_l, seq)

        return ret_log_clust

    def add_seq_to_prefix_tree(self, rn, log_clust):
        seq_len = len(log_clust.log_template)
        if seq_len not in rn.child_d:
            firt_layer_node = Node(depth=1, digit_or_token=seq_len)
            rn.child_d[seq_len] = firt_layer_node
        else:
            firt_layer_node = rn.child_d[seq_len]

        parentn = firt_layer_node

        current_depth = 1
        for token in log_clust.log_template:

            # Add current log cluster to the leaf node
            if current_depth >= self.depth or current_depth > seq_len:
                if len(parentn.child_d) == 0:
                    parentn.child_d = [log_clust]
                else:
                    parentn.child_d.append(log_clust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.child_d:
                if not self.has_numbers(token):
                    if '<*>' in parentn.child_d:
                        if len(parentn.child_d) < self.max_child:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parentn.child_d[token] = new_node
                            parentn = new_node
                        else:
                            parentn = parentn.child_d['<*>']
                    else:
                        if len(parentn.child_d) + 1 < self.max_child:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parentn.child_d[token] = new_node
                            parentn = new_node
                        elif len(parentn.child_d) + 1 == self.max_child:
                            new_node = Node(depth=current_depth + 1, digit_or_token='<*>')
                            parentn.child_d['<*>'] = new_node
                            parentn = new_node
                        else:
                            parentn = parentn.child_d['<*>']

                else:
                    if '<*>' not in parentn.child_d:
                        new_node = Node(depth=current_depth + 1, digit_or_token='<*>')
                        parentn.child_d['<*>'] = new_node
                        parentn = new_node
                    else:
                        parentn = parentn.child_d['<*>']

            # If token is matched
            else:
                parentn = parentn.child_d[token]

            current_depth += 1

    # seq1 is template
    def seq_dist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        sim_tokens = 0
        num_of_par = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                num_of_par += 1
                continue
            if token1 == token2:
                sim_tokens += 1

        ret_val = float(sim_tokens) / len(seq1)

        return ret_val, num_of_par

    def fast_match(self, log_clust_l, seq):
        ret_log_clust = None

        max_sim = -1
        max_num_of_para = -1
        max_clust = None

        for logClust in log_clust_l:
            cur_sim, cur_num_of_para = self.seq_dist(logClust.log_template, seq)
            if cur_sim > max_sim or (cur_sim == max_sim and cur_num_of_para > max_num_of_para):
                max_sim = cur_sim
                max_num_of_para = cur_num_of_para
                max_clust = logClust

        if max_sim >= self.st:
            ret_log_clust = max_clust

        return ret_log_clust

    def get_template(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        ret_val = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                ret_val.append(word)
            else:
                ret_val.append('<*>')

            i += 1

        return ret_val

    def output_result(self, log_clust_l):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for log_clust in log_clust_l:
            template_str = ' '.join(log_clust.log_template)
            occurrence = len(log_clust.log_idl)
            template_id = hashlib.md5(
                template_str.encode('utf-8')).hexdigest()[0:8]
            for log_id in log_clust.log_idl:
                log_id -= 1
                log_templates[log_id] = template_str
                log_templateids[log_id] = template_id
            df_events.append([template_id, template_str, occurrence])

        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        if self.keep_para:
            self.df_log["ParameterList"] = self.get_parameter_list

        try:
            self.df_log["timestamp"] = self.df_log["timestamp"].astype(
                'datetime64')
            time_delta = (self.df_log["timestamp"] -
                          self.df_log["timestamp"].shift(1)) / np.timedelta64(1, 's')
            self.df_log["ParameterList"] = self.df_log["ParameterList"].astype(
                str) + " ,TimeDiff: " + time_delta.astype(str)
        except ParserError:
            print("Unrecognized timestamp format")

        file = os.path.basename(self.log_name)
        file = os.path.splitext(file)[0]

        self.df_log.to_csv(os.path.join(
            self.save_path, file + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(
            lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.save_path, file + '_templates.csv'),
                        index=False, columns=["EventId", "EventTemplate", "Occurrences"])

    def print_tree(self, node, dep):
        p_str = ''
        for i in range(dep):
            p_str += '\t'

        if node.depth == 0:
            p_str += 'Root'
        elif node.depth == 1:
            p_str += '<' + str(node.digitOrtoken) + '>'
        else:
            p_str += node.digitOrtoken

        print(p_str)
        logger.info(p_str)

        if node.depth == self.depth:
            return 1
        for child in node.child_d:
            self.print_tree(node.child_d[child], dep + 1)

    def parse(self, log_name, df_log=None):
        print('Parsing: ' + os.path.join(self.path, log_name))
        logger.info('Parsing: ' + os.path.join(self.path, log_name))
        start_time = datetime.now()
        self.log_name = log_name
        root_node = Node()
        log_clu_l = []

        if df_log is None:
            if self.log_name.endswith(".csv"):
                self.csv_to_dataframe(csv_file=log_name)
            else:
                self.load_data()
        else:
            self.df_log = df_log

        count = 0
        for idx, line in self.df_log.iterrows():
            log_id = line['LineId']

            logmessage_l = self.preprocess(line['log_message']).strip().split()

            match_cluster = self.tree_search(root_node, logmessage_l)

            # Match no existing log cluster
            if match_cluster is None:
                new_cluster = Logcluster(
                    log_template=logmessage_l, log_idl=[log_id])
                log_clu_l.append(new_cluster)
                self.add_seq_to_prefix_tree(root_node, new_cluster)

            # Add the new log message to the existing cluster
            else:
                new_template = self.get_template(
                    logmessage_l, match_cluster.log_template)
                match_cluster.log_idl.append(log_id)
                if ' '.join(new_template) != ' '.join(match_cluster.log_template):
                    match_cluster.log_template = new_template

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(
                    count * 100.0 / len(self.df_log)))
                logger.info('Processed {0:.1f}% of log lines.'.format(
                    count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.output_result(log_clu_l)

        print('Parsing done. [Time taken: {!s}]'.format(
            datetime.now() - start_time))
        logger.info('Parsing successful. [Time taken: {!s}]'.format(
            datetime.now() - start_time))

        return self.df_log

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(
            self.path, self.log_name), regex, headers, self.log_format)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        '''
        Function to transform log file to dataframe
        '''

        log_messages = []
        totallines = 0
        linecount = 0
        with open(log_file, 'r', encoding="utf8") as fin:
            for line in fin.readlines():
                totallines += 1

                if re.search("DEBUG $", line):
                    continue
                if r"            " in line:
                    continue

                if r"TimeDiff" in line:
                    continue

                if r"CORE" not in line:
                    continue

                try:
                    match = regex.search(line.strip())
                    message = []

                    for header in headers:
                        message.append(match.group(header))

                    log_messages.append(message)
                    linecount += 1
                except Exception:
                    print("ERROR")
                    logger.error("ERROR")
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def csv_to_dataframe(self, csv_file):
        '''
        Converts csv file to a dataframe
        '''

        logdf = pd.read_csv(csv_file)
        linecount = len(logdf.index)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        self.df_log = logdf

    def generate_logformat_regex(self, logformat):
        '''
        Function to generate regular expression to split log messages
        '''

        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)

        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        # template_regex = re.sub(r'\\ +', r'\s+', template_regex)
        template_regex = "^" + template_regex.replace(r"\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["log_message"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(
            parameter_list, tuple) else [parameter_list]
        return parameter_list

if __name__ == '__main__':

    log_file = "BOI-JP-BBUEM1_CORE_20200608_0002_in_use.txt"
    base = os.path.basename(log_file)
    file_name = os.path.splitext(base)[0]
    print('base:',base)
    print('filename:', file_name)

    log_format = '<timestamp>+<host> - <service> <thread_id> <message_id> <structured_data> - <log_level> <log_message>'
    regex = [  # line 450
        r'((Good)?ApplicationManagementService: \w+(\(\)|) (\- |)(\[(.*?)\]|))',
        r"\{(.*?)\}",
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # ips
        r"[0-9A-Fa-f]{8}-?([0-9A-Fa-f]{4}-?){3}[0-9A-Fa-f]{12}",  # guids
        r"(s|S)\d{1,10}",  # script id
        r"settingName=.*",
        r"added settingInternalId.*",  # settinginternal id
        r"http.*(,?\]?\s?)",  # url
        r"(\=com\.).*$",
        r"(?<=Public API).*$",
        r"(?<=Checking sorted filter chain:).*$",
        r"(?<=PostUpgradeStartupInitializer::).*$",
        r"(?<=RECONCILIATION BATCH:).*$",
        r"(?<=BB2FA for BlackBerry).*$",
        r"(?<=Creating filter chain:).*$",
        r"(?<=>>>SIS Snapin ).*$",
        r"(?<=Snapin discovery:).*$",
        r"(?<=Move system App To PermStore).*$",
        r"(?<=getInvocationRegistrationRequest \- ).*$",
        r"(?<=Exchange\[).*$",
        r"(?<=TenantSyncTask:).*$",
        r"(?<=Snapin \[).*$ found in database already, not adding",
    ]

    st = 0.5
    depth = 4
    parser = Parser(log_format=log_format, depth=depth, st=st, rex=regex)
    parsed_log_file = parser.parse(log_file)
    parsed_log_file.to_csv("parsed_%s.csv" % file_name)
