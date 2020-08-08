Libraries used:
	- io
	- os
	- glob
	- boto3
	- random
	- pickle
	- bs4
	- chardet
	- sklearn
	- itertools
	- unicodedata
	- string
	- argparse
	- six
	- re
	- matplotlib.pyplot (as plt)
	- numpy (as np)
	- pandas (as pd)
	- sagemaker
	- nltk
	- torch

Sublibraries used:

	- from io: StringIO
	- from bs4: BeautifulSoup
	- from sklearn.utils: shuffle
	- from itertools: groupby
	- from six: BytesIO
	- from sagemaker: get_execution_role
	- from sagemaker.pytorch: PyTorch, PyTorchModel
	- from sagemaker.predictor: RealTimePredictor
	- from sagemaker_containers
	- from nltk.corpus: stopwords
	- from nltk.stem.porter: *
	- from torch: torch.utils.data, torch.optim (as optim)
	- torch.nn as nn, torch.nn.functional (as F)

Apart from this libraries, no startup is needed for starting the project. Only the serve, source and website folders should be in the same directory as the notebook.

This project was done with AWS Sagemaker.
