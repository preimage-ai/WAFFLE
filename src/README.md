# WAFFLE - Multimodal Floorplan Understanding in the Wild

## Setup
Create a new conda env
```
conda create -n waffle python=3.10
conda activate waffle
```
Install the requirements

```
pip install -r requirements.txt
pip install -e src/
```

Set up GCP credentials: follow https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev

Log into hugging face in order to be able to use llama-2:
```
git config --global credential.helper store
huggingface-cli login
```

Set up SAM:
Download the SAM ckpt:
```
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```


## Data setup
`data/scraped_data/with_fields.csv` contains scraped data from images in Wiki Commons under the categories 'Floorplans' and 'Architectural drawings'.

## Creating the dataset

### 1. Create the initial dataset
Run:
```
python src/build_initial_dataset.py
```

This will create an initial datset containing some of the scraped data information. The script: and for every image:
1. Downloads the images
2. Saves CLIP scores on categories set in `src/vlms/clip/categories.py`
3. Saves OCR texts extracted from the images
Outputs will be stored under `data/outputs`

### 2. Extract data using an LLM
Run `python src/llms/textual_data_extractor.py` with the following arguments:
* `--df_path`: path to df that contains the initial dataframe
* `--llama_model_name`: the name of the llama model to use, default uses the 13B model.
* `--col_name`: the name of the column you want to add to the dataframe. Make sure it has a corresponding function in `col_name_to_fn`, and a prompt with the column name in `src/llms/prompts`

If you want to run in parallel, add the follwing arguments:
* `--split_runs`: the number of times to split the dataset and run the function.
* `--run_number`: the run number for this script. Should be between 0 and split_runs - 1.

We'll start by running:
```
python src/llms/textual_data_extractor.py --df_path data/all_data.csv --col_name topic_answer --split_runs 4 --run_number 0
```
And update `--run_number` according to the run we're on.

### 3. Finetune a classifier
We can extract a small set of accurate floorplans from the data we have so far. We'll use them to train a classifier, and extract more flooeplans from the rest of the data.

The accurate dataset can be obtained as follows:
```
import pandas as pd
BUILDING_TOPICS = ['A', 'B']

df = pd.read_csv('data/all_data.csv')
small_mask = (df.topic_answer.isin(BUILDING_TOPICS)) & (df.narrow_clip_score > 0.5) & (df.top_5_cat_are_floorplan)
df[small_mask]
```

In addition, a set of images that are most likely not florrplans can be obtained as follows:
```
df[~df.topic_answer.isin(BUILDING_TOPICS)]
```
Meaning, not buildings.

`data/for_classifier` contains triaining date extracted from these two groups.
We'll fintune ViT on it to obtain a classifier, and use it to expand our dataset.

Run:
```
python src/vlms/vit/finetune_vit_on_binary_classification.py
```

This will save our dataset under `data/large_dataset.csv` using the following filtering:
```
large_mask = (df.topic_answer.isin(BUILDING_TOPICS)) & (df.wide_clip_score > 0.5) | (df.classifier_score > 0.005)
df[large_mask]
```

### 4. Extract more data using an LLM
Now that we have our dataset we can extract more data similarly to the way we did in 2.
We'll run:

```
python src/llms/textual_data_extractor.py --df_path data/large_dataset.csv --col_name building_name --split_runs 4 --run_number 0
```

We'll run the same command for the extraction of the building type, location info and legend data.
Note that for legend extraction we'll need to use a more powerful llm, like `meta-llama/Llama-2-70b-chat-hf`


### 5. Legend formatting and rendering
After extracting our data, some of it requires further formatting.
Run:
```
python src/legends/legend_formatter.py --legends_dir data/outputs/legends_outputs/{TYPE}
```
Where `TYPE` is one of: `from_caption`, `from_wiki` or `from_ocr`.

Then, you can render the images with the formatted legends and mark the legend labels on the images use the OCR texts found earlier.
Run:
```
python src/legends/legend_image_renderer.py --legends_col {COLUMN}
``` 
Where `COLUMN` is one of: `caption_legend_fn`, `wiki_legend_fn` or `ocr_legend_fn`