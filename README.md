#### To run:
1. pip install -r requirements.txt
2. python train.py

## Arguments
### General Arguments:
- **`--device`** *(default: "cuda")*:  
  Specify the device to use for training and inference. Options include `"cuda"` for GPU or `"cpu"` for CPU.

- **`--mixed_precision`**:  
  Enable mixed precision training for faster computations and reduced memory usage. Use this flag to activate mixed precision.

- **`--debug`**:  
  Enable debug mode to produce verbose logging for troubleshooting.

### Finetuning Options:  
*(These options are mutually exclusive.)*
- **`--finetune`** *(default)*:  
  Enable finetuning mode.  
- **`--no_finetune`**:  
  Disable finetuning.

### HuggingFace Dataset for Training:
- **`--hf_dataset_train`** *(default: None)*:  
  Name of the HuggingFace dataset to use for training (e.g., `"wikitext"`).  

- **`--hf_config_train`** *(default: None)*:  
  Configuration name of the HuggingFace dataset (if applicable).

- **`--hf_split_train`** *(default: "train")*:  
  Split of the HuggingFace dataset to use for training (e.g., `"train"`).

- **`--hf_field_train`** *(default: "text")*:  
  Field name in the HuggingFace dataset that contains the text data.

### HuggingFace Dataset for Testing:
- **`--hf_dataset_test`** *(default: None)*:  
  Name of the HuggingFace dataset to use for testing (e.g., `"wikitext"`).

- **`--hf_config_test`** *(default: None)*:  
  Configuration name of the HuggingFace dataset (if applicable).

- **`--hf_split_test`** *(default: "test")*:  
  Split of the HuggingFace dataset to use for testing (e.g., `"train"`, `"validation"`, `"test"`).

- **`--hf_field_test`** *(default: "text")*:  
  Field name in the HuggingFace dataset that contains the text data.

```bash
python train.py --device "cuda" --mixed_precision --hf_dataset_train "wikitext" --hf_split_train "train"
```