# Maize Yield Regression Case Study

Predicting maize yield (tonnes per hectare) from farm and weather data, and comparing five regression models: **SLR, MLR, Polynomial (degree 2), Ridge and Lasso**. The project ends with a one-page memo recommending one model to the Head of Agronomy.

The data is **synthetic**. It is generated in the notebook with a known hidden recipe, so we can check whether each model finds the real pattern.

## The hidden truth in the data

- Fertilizer has diminishing returns.
- Rainfall has a best level near 850 mm.
- Soil pH has a best level near 6.5.
- `plot_size_ha` and `farmer_age` have no effect on yield.
- `humidity_pct` is strongly correlated with rainfall (about 0.90), so it duplicates it.

## Results

Split: 80% train (640 rows), 20% test (160 rows), `random_state=0`. Cross-validation is 5-fold on the training set only. The test set is used once per model, for the final score.

```
model                cv_r2  test_r2  test_rmse  test_mae
SLR                  0.187    0.316      1.252     0.992
MLR                  0.339    0.447      1.126     0.891
Polynomial           0.933    0.928      0.405     0.312
Ridge (alpha=10)     0.933    0.929      0.403     0.312
Ridge (tuned)        0.933    0.929      0.403     0.312
Lasso (alpha=0.01)   0.938    0.932      0.394     0.308
Lasso (tuned)        0.938    0.932      0.393     0.308
```

RMSE and MAE are in t/ha.

## Key findings

- The straight-line models (SLR, MLR) stall at about 0.3 to 0.45 R2 because they cannot follow the arch-shaped effects of rainfall and soil pH.
- Adding squared and interaction terms lifts R2 to about 0.93. The gain comes from the model's shape, not from using more variables.
- Ridge and Lasso beat plain polynomial by only about 0.004 R2, within the noise of one test split. With 640 rows and 65 columns there is little overfitting to fix.
- Lasso on the original features switches off `humidity_pct` exactly (it copies rainfall) and shrinks `plot_size_ha` and `farmer_age` to near zero.
- Recommended model: tuned Lasso (degree-2 polynomial), with the caveat that its edge over plain polynomial is small.

## Models

| Model | Idea |
|---|---|
| SLR | One input (fertilizer), one straight line |
| MLR | All 10 inputs, each a straight line |
| Polynomial | MLR plus squares and pairwise products (10 inputs become 65 columns) |
| Ridge | Polynomial plus an L2 penalty that shrinks coefficients |
| Lasso | Polynomial plus an L1 penalty that can set coefficients exactly to zero |

Ridge and Lasso alpha is tuned with 5-fold CV over 50 values from 0.001 to 100. A fixed alpha (10 for Ridge, 0.01 for Lasso) is also reported.

## Repository contents

```
maize_yield_regression_full.ipynb   Everything in one notebook (start here)
01_data_and_exploration.ipynb       Generate data, explore, plots
02_slr.ipynb                        Simple linear regression
03_mlr.ipynb                        Multiple linear regression
04_polynomial.ipynb                 Polynomial regression, degree 2
05_ridge.ipynb                      Ridge, fixed and tuned alpha
06_lasso.ipynb                      Lasso, fixed and tuned alpha, coefficient interpretation
07_compare_and_memo.ipynb           Compare models, write memo.txt
code_explained_simply.docx          Every piece of code explained in plain language
```

Running the notebooks creates these files in the working folder:

```
farm_yield.csv                      Generated dataset (800 rows, 11 columns)
memo.txt                            Recommendation memo
results_*.csv, lasso_linear_coefs.csv   Saved by the split notebooks 02 to 06
```

## How to run

Requires Python 3.9 or newer.

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

**Option A: one notebook.** Open `maize_yield_regression_full.ipynb` and run all cells from top to bottom. Later cells reuse variables from earlier ones, so run them in order.

**Option B: separate notebooks.** Run them in numbered order, 01 to 07, in the same folder. Notebook 01 creates `farm_yield.csv`, and notebooks 02 to 06 each save their results for notebook 07 to read.

Results were produced with scikit-learn 1.8, pandas 3.0, numpy 2.4 and matplotlib 3.10. Other recent versions should give the same or very similar numbers. The data uses a fixed seed (`default_rng(42)`).

## Limitations

- Synthetic data from one simulated season. Real farms will have effects not modelled here, such as soil type.
- Predictions are reliable only inside the data range (fertilizer 0 to 200 kg/ha, rainfall 250 to 1500 mm).
- Before relying on the result, test on a second real season.

## Documentation

`code_explained_simply.docx` walks through the full notebook step by step, explaining each line like you are five.
