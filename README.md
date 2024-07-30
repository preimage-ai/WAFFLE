# WAFFLE


## Dataset


### Download
Download and extract all files in the following folder: [Drive Folder](https://drive.google.com/drive/folders/1scJZgjT1HtNCjqgaafHNpu9xq6Y0pP1Q?usp=drive_link).


### Organize
Create the following folder structure using the data you downloaded and extracted:


> **_📝 TODO:_** Add all extracted data here including OD results and pre-processed legends.
> 
> **_📝 TODO:_** Pre-organize the structure in the drive folder.
```
.
dataset.csv
test_countries.json
├── ...
├── data
│   ├── original_size_images
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
* `img_path`: the relative path to where the image JPG file is stored
* `building_type`: the type of the identified building
* `high_level_building_type`: the clustered type of the identified building (out of 10 options: )
* `building_name`: the name of the identified building
* `country`: the country of the identified building
* `ocr_fn`: the relative path to where the extracted OCR texts are stored
* `ocr_texts`: the extracted texts from the image, from top to bottom & left to right
* `grounded_unified_fn`: the relative path to where the grounded legends and architectural features are stored


> **_📝 TODO:_** Include all other interesting fields.


## Helper classes
> **_📝 TODO:_** Include code references for easier access to OCR, OD, legends, etc.

## Models
> **_📝 TODO:_** Include trained models: OD, segmentation, generation etc.
