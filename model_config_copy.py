#!/usr/bin/env python3

class ModelConfig:
    """
    ####################################################################################################################
    Customer controlled parameters:
    ####################################################################################################################
    - load_from: str, load training data from local or from elasticsearch, select "local" or "elasticsearch"
    - parsed_log_path: str, local path of parsed log data
    - parsed_test_path: str, local path of parsed test data
    - figure_path: str, path to save model generated figures
    - hosts: list, elasticsearch hosts
    - index_name: str, index name
    - log_lines: int, the number of log lines to be fetch from elasticsearch
    - http_auth: tuple, HTTP authentication
    - scheme: str, scheme
    - use_ssl: bool, whether using SSL
    - verify_certs: bool, whether need to verify certificates
    - timeout: int, timeout seconds
    - correlation_id: str, the correlation id rows having more than 50% rows of total dataset
    - train_model: bool, whether training model or using pre-trained model
    - tune_parameters: bool, whether performing hyper-parameter tuning
    """
    # Source configuration
    ''' replace local with elasticsearch to trigger read from es index'''
    load_from = "local"
    # load_from = "elasticsearch"

    data_column = 'EventTemplate'  # change this to log_message to trigger read from es
    log_message_column = 'log_message'
    line_id_column = "LineId"
    timestamp_column = "timestamp"
    log_level_column = "log_level"
    label_column = 'Anomaly'

    # Local configuration
    parsed_log_path = "model/dataset/training_set/kb500.csv"
    parsed_test_path = "model/dataset/testing_set/BOI-JP-BBUEM1_CORE_20200608_0002_structured1.csv"
    parsed_validation_path = "model/dataset/validation_set/df_test.csv "

    useless_columns = ['LineId', 'service', 'EventId', 'EventTemplate', 'ParameterList']
    figure_path = "model/figures"

    # Elasticsearch configuration
    hosts = [{'host': 'escsod01.es.nebula.bblabs', 'port': '9200'},
             {'host': 'escsod02.es.nebula.bblabs', 'port': '9200'},
             {'host': 'escsod03.es.nebula.bblabs', 'port': '9200'},
             {'host': 'escsod04.es.nebula.bblabs', 'port': '9200'},
             {'host': 'escsod05.es.nebula.bblabs', 'port': '9200'}]

    index_name = "optictest-4444444440-core"
    log_lines = 10000
    http_auth = ('optic_kil', '0ptic1')
    scheme = "https"
    use_ssl = True
    verify_certs = True
    ca_certs = 'model/data_preprocessing/elk_certificates/ca_chain.crt'
    timeout = 30
    correlation_id = '[{{Correlation-Id,7b5c201f-2448-4be4-ae68-3a86aa322886}{taskName,directoryGroupSyncService}}]'

    # Training mode
    train_model = False
    test_mode = True
    tune_parameters = False
    explain_prediction_demo = True
    model_saving_path = "model/results/trained_models/"
    prediction_saving_path = "model/results/"
