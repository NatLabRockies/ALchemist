# scikit-learn Backend

The **sklearn** backend in ALchemist allows you to train a Gaussian Process (GP) surrogate model using scikit-learn's `GaussianProcessRegressor`, wrapped by scikit-optimize for Bayesian optimization workflows.

---

## What is the sklearn backend?

ALchemist's sklearn backend uses [scikit-optimize (skopt)](https://scikit-optimize.github.io/) which builds on [scikit-learn](https://scikit-learn.org/)'s Gaussian Process implementation. This provides a lightweight, CPU-only option for Bayesian optimization that works well for moderate-sized datasets.

---

## Training a Model with sklearn Backend

When you select the **sklearn** backend in the Model panel (or specify `backend='sklearn'` in code), you are training a Gaussian Process model using the scikit-learn/scikit-optimize stack. The workflow and options are as follows:

### 1. Kernel Selection

You can choose from several kernel types for the GP:

- **RBF (Radial Basis Function):** Default, smooth kernel.

- **Matern:** Flexible kernel with a tunable smoothness parameter (`nu`).

- **RationalQuadratic:** Mixture of RBF kernels with varying length scales.

For the Matern kernel, you can select the `nu` parameter (0.5, 1.5, 2.5, or ∞), which controls the smoothness of the function.

> **Note:** ALchemist uses anisotropic (dimension-wise) kernels by default, so each variable can have its own learned lengthscale. This helps preserve the physical meaning of each variable and enables automatic relevance detection (ARD). For more details, see the [Kernel Deep Dive](../background/kernels.md) in the Educational Resources section.

### 2. Optimizer

You can select the optimizer used for hyperparameter tuning:

- **L-BFGS-B** (default)

- **CG**

- **BFGS**

- **TNC**

These control how the kernel hyperparameters are optimized during model fitting.

### 3. Advanced Options

Enable advanced options to customize kernel and optimizer settings. By default, kernel hyperparameters are automatically optimized.

### 4. Noise Handling

If your experimental data includes a `Noise` column, these values are used for regularization (`alpha` parameter in scikit-learn). If not, the model uses its default regularization.

### 5. One-Hot Encoding

Categorical variables are automatically one-hot encoded for compatibility with scikit-learn.

### 6. Input and Output Transforms

ALchemist supports optional input and output scaling for scikit-learn models:

- **Input scaling**: "minmax" (scale to [0,1]), "standard" (zero mean, unit variance), "robust" (median and IQR), or "none"

- **Output scaling**: "minmax", "standard", "robust", or "none"

While not applied by default like BoTorch, transforms can improve model performance for data with varying scales or outliers.

### 7. Model Training and Evaluation

- The model is trained on your current experiment data.

- Cross-validation is performed to estimate model performance (RMSE, MAE, MAPE, R²).

- Learned kernel hyperparameters are displayed after training.

- ARD lengthscales can be extracted for feature importance analysis.

---

## How It Works

- The model uses your variable space and experiment data to fit a GP regression model.

- Optional transforms are applied based on configuration.

- The trained model is used for Bayesian optimization, suggesting new experiments via acquisition functions.

- All preprocessing (encoding, transforms, noise handling) is handled automatically.

---

## References

- [scikit-learn GaussianProcessRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
- [scikit-optimize GaussianProcessRegressor documentation](https://scikit-optimize.github.io/stable/modules/generated/skopt.learning.GaussianProcessRegressor.html)

---

For a deeper explanation of kernel selection, anisotropic kernels, and ARD, see [Kernel Deep Dive](../background/kernels.md) in the Educational Resources section.

For details on using the BoTorch backend, see the next section.