# WAFFLE: Multimodal Floorplan Understanding in the Wild

This is the official repository of WAFFLE.

[[Project Website](https://tau-vailab.github.io/WAFFLE)]
> **_📝 TODO:_** link to arxiv

> **WAFFLE: Multimodal Floorplan Understanding in the Wild**<br>
> Keren Ganon*, Morris Alper*, Rachel Mikulinsky, Hadar Averbuch-Elor<br>
> Tel Aviv University<br>
>\* Denotes equal contribution


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

Finetuned models checkpoints can be found [here](https://tauex-my.sharepoint.com/:f:/g/personal/hadarelor_tauex_tau_ac_il/Ekk92mOOP8RJgLcAVphW918B_RFwh7Z5a5eDQpyXZSanVQ?e=tgUo5k), and helper inference code under [`src/helpers`](https://github.com/TAU-VAILab/WAFFLE/tree/2c1527bc27a5a7d8285a6de1684f1dc391071c5d/src/helpers). Specifically:

| Task                                   | Model | Helper class |
| -------------------------------------- | ----- | ------------ |
| Object detection for common layout components | [ft-DETR](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FWAFFLE%2Fmodels%2Fft%5Fdetr) | [`detr_inf.py`](https://github.com/TAU-VAILab/WAFFLE/blob/2c1527bc27a5a7d8285a6de1684f1dc391071c5d/src/helpers/detr_inf.py) |
| Open-Vocabulary Floorplan Segmentation | [ft-CLIPSeg](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FWAFFLE%2Fmodels%2Fft%5Fclipseg) | [`clipseg_inf.py`](https://github.com/TAU-VAILab/WAFFLE/blob/2c1527bc27a5a7d8285a6de1684f1dc391071c5d/src/helpers/clipseg_inf.py) |
| Text-Conditioned Floorplan Generation | [ft-stable-diffusion](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FWAFFLE%2Fmodels%2Fft%5Fstable%5Fdiffusion) | |
| Structure-Conditioned Floorplan Generation | [ft-controlnet-floorplan-generation](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FWAFFLE%2Fmodels%2Fft%5Fcontrolnet%5Ffloorplan%5Fgeneration) | |
| Wall Segmentation with a Diffusion Model | [ft-controlnet-wall-detection](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FWAFFLE%2Fmodels%2Fft%5Fcontrolnet%5Fwall%5Fdetection) | [`wall_detection_inf.py`](https://github.com/TAU-VAILab/WAFFLE/blob/2c1527bc27a5a7d8285a6de1684f1dc391071c5d/src/helpers/wall_detection_inf.py) |

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
## Citation

If you find this code or our data helpful in your research or work, please cite the following paper.
