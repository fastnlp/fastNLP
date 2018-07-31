# FastNLP
```
FastNLP
├── docs
│   └── quick_tutorial.md
├── fastNLP
│   ├── action
│   │   ├── action.py
│   │   ├── inference.py
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── optimizer.py
│   │   ├── README.md
│   │   ├── tester.py
│   │   └── trainer.py
│   ├── fastnlp.py
│   ├── __init__.py
│   ├── loader
│   │   ├── base_loader.py
│   │   ├── config_loader.py
│   │   ├── dataset_loader.py
│   │   ├── embed_loader.py
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── preprocess.py
│   ├── models
│   │   ├── base_model.py
│   │   ├── char_language_model.py
│   │   ├── cnn_text_classification.py
│   │   ├── __init__.py
│   │   └── sequence_modeling.py
│   ├── modules
│   │   ├── aggregation
│   │   │   ├── attention.py
│   │   │   ├── avg_pool.py
│   │   │   ├── __init__.py
│   │   │   ├── kmax_pool.py
│   │   │   ├── max_pool.py
│   │   │   └── self_attention.py
│   │   ├── decoder
│   │   │   ├── CRF.py
│   │   │   └── __init__.py
│   │   ├── encoder
│   │   │   ├── char_embedding.py
│   │   │   ├── conv_maxpool.py
│   │   │   ├── conv.py
│   │   │   ├── embedding.py
│   │   │   ├── __init__.py
│   │   │   ├── linear.py
│   │   │   ├── lstm.py
│   │   │   ├── masked_rnn.py
│   │   │   └── variational_rnn.py
│   │   ├── __init__.py
│   │   ├── interaction
│   │   │   └── __init__.py
│   │   ├── other_modules.py
│   │   └── utils.py
│   └── saver
│       ├── base_saver.py
│       ├── __init__.py
│       ├── logger.py
│       └── model_saver.py
├── LICENSE
├── README.md
├── reproduction
│   ├── Char-aware_NLM
│   │  
│   ├── CNN-sentence_classification
│   │  
│   ├── HAN-document_classification
│   │  
│   └── LSTM+self_attention_sentiment_analysis
|
├── requirements.txt
├── setup.py
└── test
    ├── data_for_tests
    │   ├── charlm.txt
    │   ├── config
    │   ├── cws_test
    │   ├── cws_train
    │   ├── people_infer.txt
    │   └── people.txt
    ├── test_charlm.py
    ├── test_cws.py
    ├── test_fastNLP.py
    ├── test_loader.py
    ├── test_seq_labeling.py
    ├── test_tester.py
    └── test_trainer.py

```
