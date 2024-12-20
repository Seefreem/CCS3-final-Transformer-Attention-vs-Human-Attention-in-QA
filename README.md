# CCS3-final-Transformer-Attention-vs-Human-Attention-in-QA
This is the repo of the course project of Computational Cognitive Science 3 at the University of Copenhagen.
Group member:  
Zhipeng Zhao and Shiling Deng (bjc154) 


# Reproduce the results
First run the following Linux command to generate human attention and Transformer Decoder attention:
```shell
python correlation.py --data_file data/WebQAmGaze/target_experiments_IS_EN.json --model google/gemma-2-2b-it
```
You may replace the model as your own models.

This python script will generate attention vectors and calculate correlation scores and save the results into four JSON files.   
Then by running the **visualize_results.ipynb** code, you are able to visualize the correlation plots.  
Note, the attention visualization is done through Latex code. The **visualize_results.ipynb**  file will generate three Latex files including colored texts. You may need to integrate the three Latex files manually. 


