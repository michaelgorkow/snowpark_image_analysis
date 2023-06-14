from snowflake.snowpark import Session
from huggingface_hub import HfApi, hf_hub_url
import requests
import io

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
FRAMEWORK_MAPPING = {"pytorch": PYTORCH_WEIGHTS_NAME, "tensorflow": TF2_WEIGHTS_NAME}

FILE_LIST_NAMES = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "dict.txt",
    "preprocessor_config.json",
    "added_tokens.json",
    "README.md",
    "spiece.model",
    "sentencepiece.bpe.model",
    "sentencepiece.bpe.vocab",
    "sentence.bpe.model",
    "bpe.codes",
    "source.spm",
    "target.spm",
    "spm.model",
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "added_tokens.json",
    "model_args.json",
    "entity_vocab.json",
    "pooling_config.json",
]

def download_huggingface_model(session: Session, repo_id: str, framework: str, stage_location: str) -> list:
    # Collect model files
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    
    # Filter files 
    files = [file for file in files if file in FILE_LIST_NAMES+[FRAMEWORK_MAPPING[framework]]]
    
    # Construct downloads urls and target locations
    urls = [hf_hub_url(repo_id=repo_id, filename=file) for file in files]
    stage_locations = ['/'.join([stage_location,repo_id,file]) for file in files]

    for file, url, stage_location in zip(files, urls, stage_locations):
        # Download file
        r = requests.get(url)
        # Upload file
        with io.BytesIO() as out:
            out.write(r.content)
            session.file.put_stream(input_stream=out, stage_location=stage_location, auto_compress=False, overwrite=True)
        print(f'Uploaded file:{file}')
    return stage_locations