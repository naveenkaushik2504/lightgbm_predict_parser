# Parser for LighGBM classifier

This is a repository to score data using a model file built using [LightGBM](https://github.com/microsoft/LightGBM) 
classifier. 

Usage:

`python LightGBM_predict_modelfile_API.py -d test.txt -m model.txt -o out.txt`

-d -- data file

-m -- model file created using training

-o -- output file to get the probability score

