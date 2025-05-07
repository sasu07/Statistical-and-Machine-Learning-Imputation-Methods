
# Addressing Missing Data Challenges in Geriatric Health Research using Advanced Imputation Techniques

**Copyright (c) 2024 Gabriel-Vasilică Sasu. All rights reserved.**


## 1. Overview

This project explores various advanced techniques for imputing missing data, a common and significant challenge in geriatric health research. The presence of missing data can lead to biased results and reduced statistical power, making effective imputation strategies crucial. This repository provides the scripts used to implement and evaluate several imputation methods, as detailed in the research paper "Addressing Missing Data Challenges in Geriatric Health Research using Advanced Imputation Techniques."

The primary goal is to compare different imputation approaches, including traditional statistical methods, machine learning-based techniques, and deep learning models, to handle missing values in datasets relevant to geriatric health. The scripts cover methods such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Bayesian Networks, Matrix Completion, Expectation-Maximization (EM), Variational Autoencoders (VAE), Generative Adversarial Imputation Nets (GAIN), and a GRU-based imputation model.

## 2. Dataset

The datasets used in this research are publicly available through Figshare. This collection includes:

*   **Original data with missing values**: Raw datasets where missing values have been artificially introduced under Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR) conditions.
*   **Imputed data (KNN and SVM)**: Datasets that have been imputed using KNN and SVM methods, provided for comparison purposes.
*   **Machine learning models**: Details or outputs related to classifiers (e.g., Random Forest, Gradient Boosting, Decision Tree) trained on both imputed and original data (referred to as post-imputed and pre-imputed models, respectively).

You can access and download the data collection from:
[https://doi.org/10.6084/m9.figshare.c.7538409.v1](https://doi.org/10.6084/m9.figshare.c.7538409.v1)

It is recommended to download the relevant CSV files (e.g., `prelucrate_2.csv` for the complete dataset, and `prelucrate_2_mcar.csv`, `prelucrate_2_mar.csv`, `prelucrate_2_mnar.csv` for datasets with missing values) and place them in a `data/` directory at the root of this project. The scripts in this repository often use relative paths like `../prelucrate_2.csv`; you may need to adjust these paths within the scripts or structure your project directory as follows:

```
project_root/

├── scripts/  
│   ├── imputation.py
│   ├── vae.py
│   └── ... (other scripts)
├── requirements.txt
└── README.md
```


## 3. Prerequisites and Dependencies

*   Python 3.7+ (though scripts were developed with Python 3.11 in mind)
*   The necessary Python libraries are listed in the `requirements.txt` file.

To install the dependencies, navigate to the project's root directory in your terminal and run:

```bash
pip install -r requirements.txt
```

It is highly recommended to use a Python virtual environment to manage dependencies for this project.

## 4. Scripts Overview

This repository contains the following Python scripts, each implementing a specific imputation technique or evaluation process:

*   `imputation.py`: Implements data imputation using a Gated Recurrent Unit (GRU) based model. It loads a dataset with missing values, preprocesses it (scaling, one-hot encoding), loads a pre-trained GRU model, and uses it to predict and fill missing values.
*   `vae.py`: Implements a Variational Autoencoder (VAE) for imputing missing data. It preprocesses data (initial mean imputation, scaling), defines and trains a VAE, and then uses the VAE to reconstruct (impute) the data.
*   `svm.py`: Performs data imputation using Support Vector Regression (SVR). It trains an SVR model for each numerical column (using complete data) and then uses these models to predict missing values in an incomplete dataset.
*   `knn.py`: Implements K-Nearest Neighbors (KNN) imputation. For each column with missing values, it uses KNeighborsRegressor trained on other columns (where the target is not missing) to predict and fill the NaNs.
*   `bayesian_networks.py`: Uses Bayesian Networks for imputation. It involves discretizing continuous variables, defining a network structure, learning parameters using Expectation-Maximization (EM), and then performing inference to impute missing values.
*   `matrix_completion.py`: Applies matrix completion techniques (IterativeSVD and SoftImpute from `fancyimpute`) to impute missing values in numerical data after normalization.
*   `expectation_maximization.py`: Implements a custom Expectation-Maximization (EM) algorithm, assuming a multivariate normal distribution, to iteratively estimate and impute missing values.
*   `gain.py`: Implements Generative Adversarial Imputation Nets (GAIN). It trains a generator to impute missing data and a discriminator to distinguish imputed from original data, saving the best generator for final imputation.
*   `evaluation_gain.py`: This script is designed to evaluate the performance of the GAIN imputation model, likely by calculating metrics such as Root Mean Squared Error (RMSE) by comparing imputed values against original (known) values for the missing entries.
*   `compatison enhance.py`: (Note: filename is `compatison enhance.py`) This script likely performs comparisons between different imputation methods or enhances the visualization/analysis of the imputation results. Its exact functionality would involve inspecting its outputs or further documentation.

Each script includes comments explaining its purpose, key steps, and important configurations. Please refer to the comments within each file for more detailed information.

## 5. How to Reproduce the Experiment

1.  **Clone the Repository**: 
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Set up Environment & Install Dependencies**:
    Create a Python virtual environment (e.g., using `venv` or `conda`) and activate it. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download and Prepare Data**:
    Download the datasets from the Figshare link provided in Section 2 ([https://doi.org/10.6084/m9.figshare.c.7538409.v1](https://doi.org/10.6084/m9.figshare.c.7538409.v1)).
    Organize the data files (e.g., `prelucrate_2.csv`, `prelucrate_2_mcar.csv`, etc.) as suggested in Section 2, typically in a `data/` subdirectory. 
    **Important**: Review the file paths used within each Python script (e.g., `file_path = '../prelucrate_2_mcar.csv'`). You may need to adjust these paths to match your directory structure. For example, if your scripts are in the root and data is in `data/`, change `../prelucrate_2_mcar.csv` to `data/prelucrate_2_mcar.csv`.

4.  **Run Imputation Scripts**:
    Execute the individual Python scripts to perform imputation using different methods. For example:
    ```bash
    python scripts/gru.py         # For GRU-based imputation
    python scripts/vae.py         # For VAE-based imputation
    python scripts/svm.py         # For SVM-based imputation
    # ... and so on for other imputation scripts.
    ```
    Most scripts will generate new CSV files containing the imputed data (e.g., `prelucrate_imputed_gru_mcar.csv`, `data_imputed_simple_vae_mnar.csv`, etc.). Check the print statements or comments at the end of each script for the output filenames.

5.  **Run Evaluation/Comparison Scripts**:
    After running the imputation scripts, you can run scripts like `evaluation_gain.py` or `compatison enhance.py` if applicable.
    ```bash
    python scripts/evaluation_gain.py
    python scripts/compatison enhance.py 
    ```
    These scripts might require specific imputed files as input, so ensure the necessary preceding imputation scripts have been run successfully.

6.  **Review Outputs**:
    Check the generated CSV files for imputed data and any output from evaluation scripts (e.g., performance metrics, plots like `gain_training_losses.png`).

## 6. Citation

If you use the dataset associated with this project, please cite it using its DOI:

*   Sasu, Gabriel-Vasilică. (2024). *A collection of data with all datasets used in this research (Addressing Missing Data Challenges in Geriatric Health Research using Advanced Imputation Techniques)* [Data set]. Figshare. [https://doi.org/10.6084/m9.figshare.c.7538409.v1](https://doi.org/10.6084/m9.figshare.c.7538409.v1)

If you use the code or findings from the associated research paper, please cite the paper accordingly (citation details for the paper should be added here once available).

## 7. License

This project is released under the MIT License. Please see the `LICENSE` file for more details (if a LICENSE file is added to the repository).

---

