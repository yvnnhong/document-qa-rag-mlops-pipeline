### Run this every time before running any code files: 
```bash
conda activate ml-env
```
### For Testing: Run the following: 
```bash
deactivate
conda activate ml-env
python --version #verify that you see Python 3.11.13 
python tests/test_rag_pipeline.py
```