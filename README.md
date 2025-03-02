# LAMAR_baselines
We compared the performance of baseline methods with LAMAR for each downstream task.  

## Prediction of splice site
We compared with RNA-FM and SpliceAI in predicting splice sites of pre-mRNAs.

## Prediction of translation efficiency  
We compared with RNA-FM and UTR-LM in predicting mRNA translation efficiency based on 5' UTR.  

## Prediction of mRNA degradation rate  
We compared with RNA-FM in predicting mRNA degradation rate based on 3' UTR.  

## Prediction of internal ribosome entry site (IRES)  
We compared with RNA-FM in predicting internal ribosome entry site (IRES).

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
