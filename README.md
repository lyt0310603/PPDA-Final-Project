# Model-Contrastive Federated Learning For NLP Task
This is the code for course final project, and rewrite the code from [MOON](https://github.com/QinbinLi/MOON/tree/main) for NLP task

# Dependencies
* PyTorch >= 1.0.0
* torchvision >= 0.2.1
* scikit-learn >= 0.23.1

## Download Word Embedding
* Glove
  ```bash
  # cd to the path of your project
  $ wget https://nlp.stanford.edu/data/glove.6B.zip
  $ unzip glove.6B.zip
  # "glove.6B.300d.txt" must be put in the root of the project
  ```

## Results save
Need create a folder "results" in the root of the project

## Parameters

| Parametes | Defult | Description |
|------------|--------|-------------|
|model_name|LSTM|Model name </br> Options: LSTM、Transformer|
|n_layers|3|Model layers|
|embedding_dim|300|Dimension of embedding layer|
|hidden_dim|300|Dimension of hidden layer|
|dropout|0.3|Dropout rate|
|projection_dim|256|Dimension of projection layer|
|n_heads|4|Number of attention heads|
|use_pretrained_embeddings|True|Whether to use pretrained word embeddings|
|freeze_embeddings|False|Whether to freeze pretrained embeddings|
|dataset|imdb|Dataset name </br> Options: imdb、ag_news、dbpedia_14、sst2、sst5、20newsgroups、trec、yelp_review、financial_phrasebank|
|max_length|350|Maximum text length|
|beta|0.5|Dirichlet distribution parameter for data partitioning|
|vocab_size|30000|Vocabulary size|
|batch_size|32|Batch size|
|lr|0.001|Learning rate|
|epochs|10|Number of local training epochs|
|seed|114514|Random seed|
|device|cuda:0|Device to run on|
|temperature|0.5|Temperature parameter for contrastive loss|
|comm_round|30|Maximum communication rounds|
|n_parties|2|Number of clients|
|alg|moon|Communication strategy </br> Options: fedavg、fedprox、moon|
|mu|1|mu parameter for FedProx/MOON|
|model_buffer_size|1|Number of historical models stored for contrastive loss|
|sample_fraction|1.0|Fraction of clients sampled per round|

## Experiment parameter setting
||max_length|batch_size|comm_round|epochs|n_parties|lr|
|---|---|---|---|---|---|---|
|Financial Phrasebank|35|64|20|8|3|0.001|
|trec|32|64|25|5|4|0.0008
|sst2|15|128|30|6|5|0.001
|sst5|30|64|30|6|5|0.0005|
|ag_news|60|64|45|10|10|0.0005

## Quickly Start
```
python main.py --comm_round 25 --epochs 5 \
        --alg 'moon' --model_name LSTM --n_parties 4 \
        --dataset trec --max_length 32 \
        --batch_size 64 --lr 0.0008