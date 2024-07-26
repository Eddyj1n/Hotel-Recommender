# Xero Assessment: Edward Jin's Submission

## Prerequisites

1. **Install R**: Ensure R version 4.3.3 or above is installed on your system to run the script. You can download and install it from [CRAN](https://cran.r-project.org/).

## Running the Script

1. **Open Terminal or Command Prompt**: Depending on your operating system, open the terminal (Linux/macOS) or command prompt (Windows).

2. **Navigate to the Project Directory**: Use the `cd` command to change the directory to where the R script is located. For example:
    ```sh
    cd path/to/xero_ejin_submission
    ```

3. **Execute the Script**: Use the `Rscript` command to run the `recommender.R` script:
    ```sh
    Rscript recommender.R
    ```

4. **Location of Predictions**: The tab-delimited file containing the predictions will be outputted to the `predictions` folder.

## Directory Structure

- `data/`: Contains the input data required to run the model.
- `predictions/`: Contains the output of the model.
- `recommender.R`: Main script to produce predictions.
- `methodology.html`: Brief report explaining the methodology and reasoning behind the approach.
- `model_scoring.R`: R script containing initial exploratory data analysis and testing candidate models. It is not required to run the main recommender.

## Package Installation

The following packages and their dependencies will be installed automatically if not already present:

- **dplyr**: 1.0.9
- **tidyr**: 1.2.0
- **recommenderlab**: 0.2-8
- **here**: 1.0.1
