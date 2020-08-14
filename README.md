# JUMBLED LETTERS CAPSTONE PROJECT
### Motivation
"Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy, it deosn't mttaer in waht oredr the ltteers in a wrod are, the olny iprmoetnt tihng is taht the frist and lsat ltteer be at the rghit pclae. The rset can be a toatl mses and you can sitll raed it wouthit porbelm. Tihs is bcuseae the huamn mnid deos not raed ervey lteter by istlef, but the wrod as a wlohe."

Since 2003, this sentence has been circulating on the internet. However, it seems that there never was a Cambridge research about it, but the general public has been debating for some time on the reason behind why we can read that particular jumbled text.

Starting from this premise, I created and deployed model using Pytorch, SageMaker and a Jupyter Notebook that tries to reconstruct the original sentence from the jumbled letter of the words, maintaining the same first and last letter of each word for the final project of the Machine Learning Engineer Nanodegree at Udacity.

This project was done with AWS Sagemaker.

### Data and Model Used
For the data I used a subset of 1,499 blogs out of 19,320 on the [Blogger Corpus]. This model consisted of an LSTM with an encoder-decoder architecture, with a Fully Connected Layer connected to the output. 

### Results
It achieved a performance of 30% of fully correct reconstructed words and 63% of right guessed letters from 20,000 words. An additional simple HTML file was created to implement it on the web.

### Libraries:

| Library | Sublibrary |
| ------ | ------ |
|io|StringIO|
|os||
|glob||
|boto3||
|random||
|pickle||
|bs4|BeautifulSoup|
|chardet||
|sklearn.utils|shuffle|
|itertools|groupby|
|unicodedata||
|string||
|argparse||
|six|BytesIO|
|re||
|matplotlib.pyplot (as plt)||
|numpy (as np)||
|pandas (as pd)||
|sagemaker|get_execution_role|
|sagemaker.pytorch| PyTorch, PyTorchModel|
|sagemaker.predictor|RealTimePredictor|
|sagemaker_containers||
|nltk.corpus|stopwords|
|nltk.stem.porter|*|
|torch|torch.utils.data|
||torch.optim (as optim)|
||torch.nn (as nn)|
||torch.functional (as F)|

[Blogger Corpus]: <http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm>

