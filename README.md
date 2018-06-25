# FastNLP
```
FastNLP
│  LICENSE
│  README.md
│  requirements.txt
│  setup.py
|
├─action      (model independent process)
│  │  action.py (base class)
│  │  README.md
│  │  tester.py (model testing, for deployment and validation)
│  │  trainer.py  (main logic for model training)
│  │  __init__.py
│  │
|
├─docs  (documentation)
│      quick_tutorial.md
│
├─loader    (file loader for all loading operations)
│   |  base_loader.py  (base class)
│   |  config_loader.py   (model-specific configuration/parameter loader)
│   |  dataset_loader.py  (data set loader, base class)
│   |  embed_loader.py    (embedding loader, base class)
│   |  __init__.py
│
├─model  (definitions of PyTorch models)
│  │  base_model.py  (base class, abstract)
│  │  char_language_model.py  (derived class, to implement abstract methods)
│  │  word_seg_model.py  
│  │  __init__.py
│  │
│
├─reproduction   (code library for paper reproduction)
│  ├─Char-aware_NLM
│  │
│  ├─CNN-sentence_classification
│  │
│  └─HAN-document_classification
│
├─saver  (file saver for all saving operations)
│      base_saver.py
│      logger.py
│      model_saver.py
│
└─tests  (unit tests, intergrating tests, system tests)
    │  test_charlm.py
    │  test_loader.py
    │  test_trainer.py
    │  test_word_seg.py
    │
    └─data_for_tests  (test data used by models)
            charlm.txt
            cws_test
            cws_train
```
