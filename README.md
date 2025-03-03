# LAMAR_baselines
We compared the performance of baseline methods with LAMAR for each downstream task.  


## Downstream tasks
| Task                                                | Baseline Method        |
| ----------------------------------------------      | ---------------------- |
| Predict splice site of pre-mRNA                     | RNA-FM, SpliceAI       |
| Predict mRNA translation efficiency based on 5'UTR  | RNA-FM, UTR-LM         |
| Predict mRNA degradation rate based on 3' UTR       | RNA-FM                 |
| Predict internal ribosome entry site (IRES)         | RNA-FM                 |


## Deploy Baseline methods
### RNA-FM
RNA-FM is a foundation language model pretrained on non-coding RNAs, used for predicting RNA 3D structure (Nature Methods, 2024).  
The github link is https://github.com/ml4bio/RNA-FM, from which we deployed the model.  
We fine-tuned RNA-FM using the trainer of transformers, so we further installed the following packages:  
```txt
transformers==4.36.2  
accelerate==0.26.1  
evaluate==0.4.1  
tokenizers==0.15.0  
datasets==2.18.0  
```
The tokenizer was developed for RNA-FM.  

### UTR-LM
UTR-LM is a foundation language model pretrained on sequences and structures of 5' UTR, used for predicting translation efficiency of mRNA based on 5' UTR (Nature Machine Intelligence, 2024).  
The script link is https://github.com/a96123155/UTR-LM, from which we deployed the model.  
We fine-tuned UTR-LM using the trainer of transformers, and installed the same packages as fine-tuning RNA-FM.  

## SpliceAI
SpliceAI is a CNN model to predict splice site from pre-mRNA sequence (Cell, 2019).  
The script link is https://github.com/Illumina/SpliceAI, from which we deployed the model.  
We directly used the trained model to predict splice site.  
