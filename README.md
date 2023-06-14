# Snowpark for Image Analysis
Demos &amp; Tutorials for Image Analysis in Snowflake using Snowpark for Python

## Requirements
- A [Snowflake Account](https://signup.snowflake.com/) 
- [Anaconda Integration enabled by ORGADMIN](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-packages#getting-started)

## Setup
First, clone the source code for this repo to your local environment:
```bash
git clone https://github.com/michaelgorkow/snowpark_image_analysis
cd snowpark_image_analysis
```

If you are using [Anaconda](https://www.anaconda.com/products/distribution) on your local machine, create a conda env from the provided environment yml-file:
```bash
conda env create -f conda_environment.yml
conda activate py_snowpark_3_9_1_5
```
Conda will automatically install `snowflake-snowpark-python` and all other dependencies for you.

Now, launch Jupyter Notebook on your local machine:
```bash
jupyter notebook
```

## Blog Articles
There is a whole series of blog articles that reference the code in this Github repository:  
[Part 1: Uploading, Querying and Visualizing Images](https://medium.com/@michaelgorkow/image-data-in-snowflake-2d0e87924c61)  
[Part 2: Image Classification](https://medium.com/@michaelgorkow/image-data-in-snowflake-2d0e87924c61)  
[Part 3: Object Detection]()  
[Part 4: Image Embeddings]()  