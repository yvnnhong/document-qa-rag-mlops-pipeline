### Run this every time before running any code files: 
```bash
conda activate ml-env
```
### For Testing: Run the following (first, ensure that you are in the project root): 
```bash
deactivate
conda activate ml-env
python --version #verify that you see Python 3.11.13 
python src/tests/test_rag_pipeline.py
```

### Quick Note for Running: 
If you see:
```bash
(base) #→ run conda activate ml-env
(ml-env) #→ you're good, just run the test
(.venv) #→ run deactivate first, then conda activate ml-env

```