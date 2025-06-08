## How to run
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
jupyter notebook \
    --notebook-dir="." \
    --ip=0.0.0.0 --port=3225
```
```
python -m debugpy --listen 4444 test.py
```

```
fastapi dev main.py
```
