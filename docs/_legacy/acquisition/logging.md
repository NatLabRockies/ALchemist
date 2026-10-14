# Logging & Tracking

ALchemist provides robust logging and notification features to help you keep track of your optimization workflow, model settings, and acquisition strategies. This is essential for reproducibility, transparency, and effective reporting in scientific research.

---

## Notifications: Immediate Feedback for Each Acquisition

Whenever you execute an acquisition strategy (e.g., suggest the next experiment), a **notification window** will automatically pop up. This window provides a detailed summary of the result, including:

- **Point Details:**  
  The coordinates of the suggested next experiment (or batch of experiments), along with predicted values, uncertainties, and 95% confidence intervals.

- **Model Info:**  
  Information about the trained model, including backend (BoTorch or scikit-optimize), kernel type, learned hyperparameters, and recent performance metrics (e.g., RMSE, R²).

- **Strategy Info:**  
  Details about the acquisition strategy used, including the type (e.g., Expected Improvement, qEI), optimization goal (maximize/minimize), and any relevant parameters.

- **Export Options:**  
  You can export all of this information to a CSV file for record-keeping, publication, or further analysis.

This notification system ensures you always have a clear record of what was suggested and why.

---

## Logging: Keeping a Complete Experiment Record

ALchemist includes an **experiment logger** that tracks every key step in your workflow:

- **Model Training:**  
  Logs backend, kernel, hyperparameters, and performance metrics each time you train or update your model.

- **Acquisition Strategies:**  
  Logs the details of every acquisition function execution, including the strategy, parameters, suggested points, predicted values, and uncertainties.

- **Experiment Data:**  
  Logs the current state of your experimental dataset, including variable names and summary statistics.

- **Exported Logs:**  
  All logs are saved in timestamped files in the `logs/` directory, making it easy to revisit or share your workflow history.

---

## Best Practices

- **Log Every Iteration:**  
  After each acquisition and model update, use the export and logging features to save your results. This ensures you can always trace back which model, kernel, and acquisition strategy led to each experimental suggestion.

- **Reproducibility:**  
  Keeping detailed logs allows you (and others) to reproduce your optimization process, which is essential for scientific rigor and publication.

- **Transparency:**  
  By exporting notifications and logs, you can clearly communicate your workflow, decisions, and results to collaborators, reviewers, or future users.

---

## Summary

ALchemist’s notification and logging system is designed to make your active learning workflow transparent, reproducible, and easy to document. Make it a habit to export and log your results at each step—this will pay dividends when you need to report, troubleshoot, or publish your work.