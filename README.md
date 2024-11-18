# WAFFLE


## Dataset


### Download
Download and extract all files in the [following folder](https://tauex-my.sharepoint.com/:f:/g/personal/hadarelor_tauex_tau_ac_il/EqMX9nRbJ9xFiK7dR_m07b8BldS2saoZ4-ockqncJb_Hrg?e=zGIuos).


### Organize
Create the following folder structure using the data you downloaded and extracted:
```
.
dataset.csv
test_countries.json
├── ...
├── data
│   ├── original_size_images
│   ├── svg_files
│   ├── outputs
|   |   ├── ocr_outputs_v2
|   |   ├── legend_outputs
|   |   |   ├── unified_grounded
│   └── ...
└── ...
```


### Access
The following fields exist in the `dataset.csv` data frame:
* `page_id`: a unique ID associated with each entry
* `img_url`: the link to the image's associated wiki-commons page
* `svg_url`: the link to the svg's associated wiki-commons page (when it exists)
* `img_path`: the relative path to where the image JPG file is stored
* `svg_path`: the relative path to where the SVG file is stored  (when it exists)
* `building_type`: the type of the identified building
* `high_level_building_type`: the clustered type of the identified building (out of 10 options: )
* `building_name`: the name of the identified building
* `country`: the country of the identified building
* `ocr_fn`: the relative path to where the extracted OCR texts are stored
* `ocr_texts`: the extracted texts from the image, from top to bottom & left to right
* `grounded_legend_fn`: the relative path to where the grounded legends and architectural features are stored

### Benchmark for Semantic Segmentation
SVGs and PNGs can be found [here](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FWAFFLE%2Fdata%2Fbenchmark).
## Finetuned Models

Finetuned models checkpoints can be found [here](https://tauex-my.sharepoint.com/:f:/g/personal/hadarelor_tauex_tau_ac_il/Ekk92mOOP8RJgLcAVphW918B_RFwh7Z5a5eDQpyXZSanVQ?e=tgUo5k), and helper inference code under `src/helpers`.

## Code

All the code for creating the dataset and finetuning the models is under `src`. Some of the funtuning code requires additional training data which can be found [here](https://tauex-my.sharepoint.com/:f:/g/personal/hadarelor_tauex_tau_ac_il/Ej-L4PUWuf9Bpg4GNz2PffIByQfcVhiubBaO_WwLZ52QYw?e=2owToB). The code should be run in the following environment:

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
