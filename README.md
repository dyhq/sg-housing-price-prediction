# Singapore Housing Price Prediction
This project aims to predict housing prices in Singapore and provide insights on factors affecting inflation.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/dyhq/sg-housing-price-prediction
   cd singapore-housing-price-prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages (see Dependencies below).
   
## Usage

1. Ensure data files are extracted to Kaggle_HDB folder in the project root directory.

2. Run the script:
   ```
   python sg-housing-price.py
   ```

3. The report can be found in `Reporting.pdf`.

## File Structure

- `sg-housing-price.py`: Handles data loading and preprocessing, performs EDA and generates visualizations, builds and tunes the Random Forest model
- `Reporting.pdf`: Contains the final report with insights and recommendations

## Dependencies

- python=3.10.15
- pandas=2.2.3
- matplotlib=3.9.2
- seaborn=0.13.2
- scikit-learn=1.5.2
- scikit-optimize=0.10.2
- numpy=2.1.1
- joblib=1.4.2
