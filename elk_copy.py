import json
import random
import warnings
import pandas as pd
import time

from elasticsearch import Elasticsearch
from elasticsearch import exceptions
from elasticsearch_dsl import Search

from model_config_copy import ModelConfig

random.seed(2020)
warnings.filterwarnings(action='ignore')


class ELKConnector:
    """
    Description: This class is used for querying log data from Elasticsearch.
    """

    def __init__(self, hosts, index_name, log_lines):
        self.hosts = hosts
        self.index_name = index_name
        self.log_lines = log_lines

    def log_time(self, message):
        print(message + ': ', time.ctime(time.time()))

    # Create Elasticsearch client
    def client_creating(self, http_auth, scheme, use_ssl, verify_certs, ca_certs, timeout):
        """
        :param http_auth: tuple, the tuple stores http authentication
        :param scheme: str, scheme ("http" or "https")
        :param use_ssl: bool, whether use SSL for connection
        :param verify_certs: bool, whether need to verify certification
        :param ca_certs: str, path of certification
        :param timeout: int, timeout
        :return:
            client: object, Elasticsearch client
        """
        self.log_time("Connection handshake start")
        client = Elasticsearch(
            [host['host'] for host in self.hosts],
            http_auth=http_auth,
            scheme=scheme,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ca_certs=ca_certs,
            timeout=timeout
        )
        try:
            # use the JSON library's dump() method for indentation
            info = json.dumps(client.info(), indent=4)
            # pass client object to info() method
            print("Elasticsearch client info():", info)
        except exceptions.ConnectionError as err:
            # print ConnectionError for Elasticsearch
            print("\nElasticsearch info() ERROR:", err)
            print("\nThe client host:", self.hosts, "is invalid or cluster is not running")
            # change the client's value to 'None' if ConnectionError
            client = None
        self.log_time("Connection established")
        return client

    # Query log data from Elasticsearch
    def log_querying(self, client,
                     included_columns=['@timestamp', 'logLevel', 'logMessage'],
                     rename_dict=None):
        """
        :param client: object, Elasticsearch client
        :param included_columns: list, columns to include in search
        :param rename_dict: dict, columns to rename in search
        :return:
            log_data: dataframe (pandas), contains log data queried from Elasticsearch
        """
        if rename_dict is None:
            rename_dict = {'logMessage': ModelConfig.log_message_column,
                           '@timestamp': ModelConfig.timestamp_column,
                           'logLevel': ModelConfig.log_level_column}

        self.log_time("Log querying start")
        log_data = pd.DataFrame()
        doc_count = 0

        query_results = Search(using=client, index=self.index_name).query("match", service="core").params(
            preserve_order=True).extra(_source={'includes': included_columns})

        for hit in query_results.scan():
            log_data = log_data.append(hit.to_dict(), ignore_index=True)
            doc_count += 1
            if doc_count % 10000 == 0:
                print(doc_count)
            if doc_count >= self.log_lines:
                break

        try:
            # rename columns
            log_data.rename(columns=rename_dict, inplace=True)
        except:
            print("Error in finding \"log_message\" field")

        # Removing duplicates
        log_data = log_data.astype(str).drop_duplicates(subset=[ModelConfig.timestamp_column,
                                                                ModelConfig.log_message_column],
                                                        keep='first').reset_index(drop=True)

        # log_data = log_data[[ModelConfig.timestamp_column,
        #                      ModelConfig.log_level_column,
        #                      ModelConfig.log_message_column]]

        self.log_time("Log querying end")
        return log_data
