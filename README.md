# Synthetic Table Generation

This repository contains the code required to generate synthetic data tables based on the content extracted from real tables. The code has been adapted from hassan-mahmood/TIES_DataGeneration.

## Installation

* Install the requirements
```
pip install requirements.txt
```
* [Download](https://github.com/mozilla/geckodriver/releases) and unpack the latest geckodriver executable for your system, ensuring the executable is added to your system `PATH`. This can be done using the following command:
```
export PATH=$PATH:/path/to/directory/of/downloaded/geckodriver/executable
```

## Usage

To run the code using the default parameters and paths, use the following command, which should work out of the box:
```
python generate_data.py
```
For a list of the arguments and their descriptions and default values, use:
```
python generate_data.py --help
```

### Inputs

The code generates synthetic tables based on content extracted from the UNLV dataset, which includes alphabetical words, numbers and special characters. Content extraction has already been performed, and the content is stored in the binary file `unlv_distribution` in this repository.

If you wish to extract the content from an alternative set of tables, the functionality is provided within this code to generate an equivalent binary file, but note this has not been tested by the S2DS team.

To do this you must specify the paths for the table images (`--imagespath`), OCR ground truths (`--ocrpath`) and table ground truths (`--tablepath`).

To understand the structure and format of these files, you can download the UNLV dataset [here](https://drive.google.com/drive/folders/1yES8Se8pyGsvLt92dJFz7z7AJQHjt4GA?usp=sharing), which contains all the input files used to generate the `unlv_distribution`, however this is not required to run the code.

### Outputs

Based on the distribution of words, the code generates 4 categories of table:

* Category 1 are plain images with no merging and with ruling lines
* Category 2 adds different border types including occasional absence of ruling lines
* Category 3 introduces cell and column merging
* Category 4 adds linear perspective transform (rotation and shear transformations) to model images captured via camera or poorly scanned

The code outputs the following files:

#### Data
* Table entities and their bounding box (bbox) positions, stored as pickled lists in `output/data/category{1-4}`
* Table cell, row and column adjacency matrices stored as pickled dictionaries in `output/data/category{1-4}`

#### Images
* Raw table images stored in `output/images/category{1-4}/raw`
* Table images with text bboxes drawn, colour-coded based of whether the entities share a cell, row or column. These are stored in `output/images/category{1-4}/bboxes` with `_cell.jpg`, `_row.jpg` and `_col.jpg` extensions, respectively
* Tables as HTML files, stored in `output/images/category{1-4}/html`
* Tables as CSV files, stored in `output/images/category{1-4}/csv`


## Table Generation
A table is generated in multiple steps, with each step contributing to generation of table):

1. The data types of columns are defined i.e. which column will contain words, numbers or special characters
2. Some cells are randomly selected for missing data
3. Rows and column spans are added to table
4. The table can be categorised in two ways based on headers(both categories are equally likely to be chosen):
    -   Table with regular headers (only the first row contains headers)
    -   Table with irregular headers (headers in first row and first column, with the possibility of multiple row spans for headers in the first column)
5. Table borders are chosen randomly, with 4 possibilities (all 4 categories are equally likely to be chosen):
    -   All borders
    -   No borders
    -   Borders only under headings
    -   Only internal borders
6. An equivalent HTML code is generated for this table.
7. This HTML coded table is converted to image using selenium.
6. Finally, shear and rotation transformations are applied to the table image.
