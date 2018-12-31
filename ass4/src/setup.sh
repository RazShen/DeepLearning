#!/bin/sh
sudo apt-get install curl
if [ ! -f glove.6B.200d.txt ]; then
	echo "Downloading pretrained word vectors:"
	curl -S https://nlp.stanford.edu/data/glove.6B.zip > all_gloves.zip
	unzip all_gloves.zip
	rm glove.6B.50d.txt glove.6B.100d.txt glove.6B.300d.txt all_gloves.zip
else
	echo "pretrained word vectors file already exists"
fi

if [ ! -d snli_1.0 ]; then 
	echo "Downloading train, dev and test files:"
	curl -S https://nlp.stanford.edu/projects/snli/snli_1.0.zip > snli_1.0.zip
	unzip snli_1.0.zip
else
	echo "snli_1.0 files already exists"
fi

echo "finishing obtaining data"

