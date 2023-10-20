# Deepfake Detection 🎭

## Pretrained Model 📋
- Download pretrained model from [here](https://drive.google.com/drive/folders/1MiF_PBXYCfAi8UNkenpqrBXzfq4bOC_e)
- Put them to `app/ml/assets/checkpoints/`

## Install ⚙️
Firstly, you have to create `virtual enviroment` (if not have) by following below command:
```shell
python -m venv deepfake_env
```
Activate your virtual environment by below command:
```shell
# On Windows:
deepfake_env\Scripts\activate

# On Unix or MacOS:
source deepfake_env/bin/activate
```

If you use GPU, install by following shell:
```shell
pip install -r requirements_gpu.txt
```
If you don't any GPU, install by following shell:
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

## Run tests 🔨
Tests for this project are defined in the `tests/` folder.
```shell
pytest app/tests/test_api
```

## Run pylint ✒️
Check score for code syntax of `folder` or `filename`.
```shell
pylint ./app/{folder}/{filename}
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

