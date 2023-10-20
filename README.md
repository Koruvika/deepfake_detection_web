# Deepfake Detection 🎭

## Pretrained Model 📋
- Download pretrained model from [here](https://drive.google.com/drive/folders/1MiF_PBXYCfAi8UNkenpqrBXzfq4bOC_e)
- Put them to `app/ml/assets/checkpoints/`

## Install ⚙️
If you use gpu, install by following shell:
```shell
pip install -r requirements_gpu.txt
```
If you don't  any GPU, install by following shell:
```shell
pip install -r requirements.txt
```

## Run demo 🚀
At first, you must run FastAPI server by below command:
```shell
uvicorn app.main:app --reload
```
After that, open another terminal and run this command below for Streamlit Application:
```shell
streamlit run app/frontend/run.py
```

## Tree directory 📁
~~~
app
├── api                  - web related stuff.
│   ├── errors           - definition of error handlers.
│   ├── routes           - web routes.
│   ├── services         - logic that is not just crud related.
│   └── responses        - response for api request corresponding.
├── ml                   - ML model and others.
│   ├── assets           - checkpoint, configs of ML models.
│   └── base_model       - base Deepfake Detection model. 
├── core                 - application configuration, startup events, logging, custom prompt.
├── logger               - export log for server process.
├── tests                - test api, code.
├── frontend             - Streamlit UI.
├── resources            - image, audio, csv, etc. (ignore)
└── main.py              - FastAPI application creation and configuration.
~~~

