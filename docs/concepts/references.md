# References

This is the reading list behind the Concepts pages. Garnett (2023), *Bayesian Optimization*, is the primary text: the Concepts pages cite it by section and equation number, and it is free to read at [bayesoptbook.com](https://bayesoptbook.com). The other works below are grouped by topic, each with a one-line note on why it matters to ALchemist.

Each entry has a stable anchor of the form `references.md#surname-year`, built from the lowercase first-author surname and the year. A letter suffix separates the two Gneiting 2007 papers. The Box family uses the co-authors' surnames as well (`box-hunter-1957`, `box-hunter-1961`) because Box wrote several papers with different co-authors.

## BO and GPs

### Garnett (2023) { #garnett-2023 }

Garnett, R. (2023). *Bayesian Optimization*. Cambridge University Press. [doi:10.1017/9781108348973](https://doi.org/10.1017/9781108348973); free at [bayesoptbook.com](https://bayesoptbook.com).

The primary text: GP modeling, policies, implementation and extensions.

### Rasmussen and Williams (2006) { #rasmussen-2006 }

Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [doi:10.7551/mitpress/3206.001.0001](https://doi.org/10.7551/mitpress/3206.001.0001); free at [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/).

The GP posterior, kernels (section 4.2), automatic relevance determination (section 5.1) and leave-one-out cross-validation (section 5.4.2).

### Shahriari et al. (2016) { #shahriari-2016 }

Shahriari, B. et al. (2016). Taking the human out of the loop: a review of Bayesian optimization. *Proceedings of the IEEE* 104(1):148-175. [doi:10.1109/JPROC.2015.2494218](https://doi.org/10.1109/JPROC.2015.2494218).

A concise survey of the field.

### Frazier (2018) { #frazier-2018 }

Frazier, P. I. (2018). A tutorial on Bayesian optimization. [arXiv:1807.02811](https://arxiv.org/abs/1807.02811).

A short derivation of expected improvement and the knowledge gradient.

### Snoek et al. (2012) { #snoek-2012 }

Snoek, J., Larochelle, H. and Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems* 25. [arXiv:1206.2944](https://arxiv.org/abs/1206.2944).

The Matern 5/2 default kernel and hyperparameter marginalization.

### Stein (1999) { #stein-1999 }

Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer. [doi:10.1007/978-1-4612-1494-6](https://doi.org/10.1007/978-1-4612-1494-6).

The case for the Matern kernel over the RBF kernel.

### Balandat et al. (2020) { #balandat-2020 }

Balandat, M. et al. (2020). BoTorch: a framework for efficient Monte-Carlo Bayesian optimization. *Advances in Neural Information Processing Systems* 33. [arXiv:1910.06403](https://arxiv.org/abs/1910.06403).

The library behind ALchemist's BoTorch backend.

### Gardner et al. (2018) { #gardner-2018 }

Gardner, J. R. et al. (2018). GPyTorch: blackbox matrix-matrix Gaussian process inference with GPU acceleration. *Advances in Neural Information Processing Systems* 31. [arXiv:1809.11165](https://arxiv.org/abs/1809.11165).

The source of the kernels and likelihoods used by the BoTorch backend.

### Hvarfner et al. (2024) { #hvarfner-2024 }

Hvarfner, C., Hellsten, E. O. and Nardi, L. (2024). Vanilla Bayesian optimization performs great in high dimensions. *International Conference on Machine Learning*. [arXiv:2402.02229](https://arxiv.org/abs/2402.02229).

Why BoTorch's default `SingleTaskGP` is now an RBF kernel with dimension-scaled priors.

### Lee et al. (2018) { #lee-2018 }

Lee, J. et al. (2018). Deep neural networks as Gaussian processes. *International Conference on Learning Representations*. [arXiv:1711.00165](https://arxiv.org/abs/1711.00165).

The theory behind the infinite-width network kernel.

### Li et al. (2023) { #li-2023 }

Li, Y. L., Rudner, T. G. J. and Wilson, A. G. (2023). A study of Bayesian neural network surrogates for Bayesian optimization. [arXiv:2305.20028](https://arxiv.org/abs/2305.20028).

The IBNN kernel used for Bayesian optimization.

### Shields et al. (2021) { #shields-2021 }

Shields, B. J. et al. (2021). Bayesian reaction optimization as a tool for chemical synthesis. *Nature* 590:89-96. [doi:10.1038/s41586-021-03213-y](https://doi.org/10.1038/s41586-021-03213-y).

The chemistry use case, including categorical reagents.

### Hase et al. (2018) { #hase-2018 }

Häse, F. et al. (2018). Phoenics: a Bayesian optimizer for chemistry. *ACS Central Science* 4(9):1134-1145. [doi:10.1021/acscentsci.8b00307](https://doi.org/10.1021/acscentsci.8b00307).

An early Bayesian optimizer built for chemistry problems.

### Hase et al. (2021) { #hase-2021 }

Häse, F. et al. (2021). Gryffin: an algorithm for Bayesian optimization of categorical variables informed by expert knowledge. *Applied Physics Reviews* 8:031406. [doi:10.1063/5.0048164](https://doi.org/10.1063/5.0048164).

Categorical Bayesian optimization for chemistry.

## Acquisition functions

### Kushner (1964) { #kushner-1964 }

Kushner, H. J. (1964). A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. *Journal of Basic Engineering* 86(1):97-106. [doi:10.1115/1.3653121](https://doi.org/10.1115/1.3653121).

The origin of probability of improvement.

### Jones et al. (1998) { #jones-1998 }

Jones, D. R., Schonlau, M. and Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization* 13:455-492. [doi:10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147).

Expected improvement, plus cross-validated standardized residuals as a GP diagnostic.

### Srinivas et al. (2010) { #srinivas-2010 }

Srinivas, N. et al. (2010). Gaussian process optimization in the bandit setting: no regret and experimental design. *International Conference on Machine Learning*. [arXiv:0912.3995](https://arxiv.org/abs/0912.3995). Journal version: *IEEE Transactions on Information Theory* 58(5):3250-3265 (2012), [doi:10.1109/TIT.2011.2182033](https://doi.org/10.1109/TIT.2011.2182033).

GP-UCB and the exploration weight β<sub>t</sub>.

### Ament et al. (2023) { #ament-2023 }

Ament, S. et al. (2023). Unexpected improvements to expected improvement for Bayesian optimization. *Advances in Neural Information Processing Systems* 36. [arXiv:2310.20708](https://arxiv.org/abs/2310.20708).

LogEI, which avoids the vanishing gradients of plain expected improvement.

### Letham et al. (2019) { #letham-2019 }

Letham, B. et al. (2019). Constrained Bayesian optimization with noisy experiments. *Bayesian Analysis* 14(2):495-519. [doi:10.1214/18-BA1110](https://doi.org/10.1214/18-BA1110).

Noisy expected improvement.

### Picheny et al. (2013) { #picheny-2013 }

Picheny, V., Wagner, T. and Ginsbourger, D. (2013). A benchmark of kriging-based infill criteria for noisy optimization. *Structural and Multidisciplinary Optimization* 48(3):607-626. [doi:10.1007/s00158-013-0919-4](https://doi.org/10.1007/s00158-013-0919-4).

How to choose the incumbent under noise.

### Brochu et al. (2011) { #brochu-2011 }

Brochu, E., Hoffman, M. W. and de Freitas, N. (2011). Portfolio allocation for Bayesian optimization. *Uncertainty in Artificial Intelligence*. [arXiv:1009.5419](https://arxiv.org/abs/1009.5419).

GP-Hedge, the portfolio acquisition strategy.

## Batch, multi-objective and constrained BO

### Ginsbourger et al. (2010) { #ginsbourger-2010 }

Ginsbourger, D., Le Riche, R. and Carraro, L. (2010). Kriging is well-suited to parallelize optimization. In *Computational Intelligence in Expensive Optimization Problems*, Springer. [doi:10.1007/978-3-642-10701-6_6](https://doi.org/10.1007/978-3-642-10701-6_6).

The q-point expected improvement for batch selection.

### Wilson et al. (2018) { #wilson-2018 }

Wilson, J. T., Hutter, F. and Deisenroth, M. P. (2018). Maximizing acquisition functions for Bayesian optimization. *Advances in Neural Information Processing Systems* 31. [arXiv:1805.10196](https://arxiv.org/abs/1805.10196).

Monte-Carlo acquisition functions and the reparameterization trick.

### Daulton et al. (2020) { #daulton-2020 }

Daulton, S., Balandat, M. and Bakshy, E. (2020). Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization. *Advances in Neural Information Processing Systems* 33. [arXiv:2006.05078](https://arxiv.org/abs/2006.05078).

The qEHVI acquisition function.

### Daulton et al. (2021) { #daulton-2021 }

Daulton, S., Balandat, M. and Bakshy, E. (2021). Parallel Bayesian optimization of multiple noisy objectives with expected hypervolume improvement. *Advances in Neural Information Processing Systems* 34. [arXiv:2105.08195](https://arxiv.org/abs/2105.08195).

The qNEHVI acquisition function, which handles noisy observations.

### Emmerich et al. (2006) { #emmerich-2006 }

Emmerich, M. T. M., Giannakoglou, K. C. and Naujoks, B. (2006). Single- and multiobjective evolutionary optimization assisted by Gaussian random field metamodels. *IEEE Transactions on Evolutionary Computation* 10(4):421-439. [doi:10.1109/TEVC.2005.859463](https://doi.org/10.1109/TEVC.2005.859463).

Expected hypervolume improvement (EHVI).

### Zitzler and Thiele (1999) { #zitzler-1999 }

Zitzler, E. and Thiele, L. (1999). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. *IEEE Transactions on Evolutionary Computation* 3(4):257-271. [doi:10.1109/4235.797969](https://doi.org/10.1109/4235.797969).

The hypervolume indicator.

### Gelbart et al. (2014) { #gelbart-2014 }

Gelbart, M. A., Snoek, J. and Adams, R. P. (2014). Bayesian optimization with unknown constraints. *Uncertainty in Artificial Intelligence*. [arXiv:1403.5607](https://arxiv.org/abs/1403.5607).

Output constraints, in contrast with ALchemist's linear input constraints.

## Classical DoE and RSM

### Montgomery (2017) { #montgomery-2017 }

Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th ed. Wiley.

Factorial designs (chapter 6), fractional factorials and resolution (chapter 8) and response surface designs (chapter 11).

### Box, Hunter and Hunter (2005) { #box-hunter-hunter-2005 }

Box, G. E. P., Hunter, J. S. and Hunter, W. G. (2005). *Statistics for Experimenters*, 2nd ed. Wiley.

A standard textbook on experimental design.

### Myers et al. (2016) { #myers-2016 }

Myers, R. H., Montgomery, D. C. and Anderson-Cook, C. M. (2016). *Response Surface Methodology*, 4th ed. Wiley.

The choice between central composite and Box-Behnken designs, the axial distance α, and center points.

### Box and Wilson (1951) { #box-wilson-1951 }

Box, G. E. P. and Wilson, K. B. (1951). On the experimental attainment of optimum conditions. *Journal of the Royal Statistical Society B* 13(1):1-45. [doi:10.1111/j.2517-6161.1951.tb00067.x](https://doi.org/10.1111/j.2517-6161.1951.tb00067.x).

The origin of the central composite design.

### Box and Hunter (1957) { #box-hunter-1957 }

Box, G. E. P. and Hunter, J. S. (1957). Multi-factor experimental designs for exploring response surfaces. *Annals of Mathematical Statistics* 28(1):195-241. [doi:10.1214/aoms/1177707047](https://doi.org/10.1214/aoms/1177707047).

Rotatability.

### Box and Hunter (1961) { #box-hunter-1961 }

Box, G. E. P. and Hunter, J. S. (1961). The 2<sup>k−p</sup> fractional factorial designs, part I. *Technometrics* 3(3):311-351. [doi:10.1080/00401706.1961.10489951](https://doi.org/10.1080/00401706.1961.10489951).

Design resolution.

### Box and Behnken (1960) { #box-behnken-1960 }

Box, G. E. P. and Behnken, D. W. (1960). Some new three level designs for the study of quantitative variables. *Technometrics* 2(4):455-475. [doi:10.1080/00401706.1960.10489912](https://doi.org/10.1080/00401706.1960.10489912).

The Box-Behnken design.

### Plackett and Burman (1946) { #plackett-1946 }

Plackett, R. L. and Burman, J. P. (1946). The design of optimum multifactorial experiments. *Biometrika* 33(4):305-325. [doi:10.1093/biomet/33.4.305](https://doi.org/10.1093/biomet/33.4.305).

The Plackett-Burman screening designs.

### Surowiec et al. (2017) { #surowiec-2017 }

Surowiec, I. et al. (2017). Generalized subset designs in analytical chemistry. *Analytical Chemistry* 89(12):6491-6497. [doi:10.1021/acs.analchem.7b00506](https://doi.org/10.1021/acs.analchem.7b00506).

Generalized subset designs (GSD).

## Optimal design

### Atkinson et al. (2007) { #atkinson-2007 }

Atkinson, A. C., Donev, A. N. and Tobias, R. D. (2007). *Optimum Experimental Designs, with SAS*. Oxford University Press. [doi:10.1093/oso/9780199296590.001.0001](https://doi.org/10.1093/oso/9780199296590.001.0001).

The D-, A- and I-criteria and exchange algorithms.

### Goos and Jones (2011) { #goos-2011 }

Goos, P. and Jones, B. (2011). *Optimal Design of Experiments: A Case Study Approach*. Wiley. [doi:10.1002/9781119974017](https://doi.org/10.1002/9781119974017).

A practitioner's text covering D versus I optimality, efficiency and categorical factors.

### Fedorov (1972) { #fedorov-1972 }

Fedorov, V. V. (1972). *Theory of Optimal Experiments*. Academic Press.

The classical theory of optimal experimental design.

### Mitchell (1974) { #mitchell-1974 }

Mitchell, T. J. (1974). An algorithm for the construction of "D-optimal" experimental designs. *Technometrics* 16(2):203-210. [doi:10.1080/00401706.1974.10489175](https://doi.org/10.1080/00401706.1974.10489175).

The DETMAX exchange algorithm.

### Meyer and Nachtsheim (1995) { #meyer-1995 }

Meyer, R. K. and Nachtsheim, C. J. (1995). The coordinate-exchange algorithm for constructing exact optimal experimental designs. *Technometrics* 37(1):60-69. [doi:10.1080/00401706.1995.10485889](https://doi.org/10.1080/00401706.1995.10485889).

The coordinate-exchange algorithm.

### Wynn (1970) { #wynn-1970 }

Wynn, H. P. (1970). The sequential generation of D-optimum experimental designs. *Annals of Mathematical Statistics* 41(5):1655-1664. [doi:10.1214/aoms/1177696809](https://doi.org/10.1214/aoms/1177696809).

A source for ALchemist's `sequential` and `simple_exchange` algorithms.

### Dykstra (1971) { #dykstra-1971 }

Dykstra, O. (1971). The augmentation of experimental data to maximize |X′X|. *Technometrics* 13(3):682-688. [doi:10.1080/00401706.1971.10488830](https://doi.org/10.1080/00401706.1971.10488830).

A source for ALchemist's `sequential` and `simple_exchange` algorithms.

## Space-filling designs

### McKay et al. (1979) { #mckay-1979 }

McKay, M. D., Beckman, R. J. and Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics* 21(2):239-245. [doi:10.1080/00401706.1979.10489755](https://doi.org/10.1080/00401706.1979.10489755).

Latin hypercube sampling.

### Sobol (1967) { #sobol-1967 }

Sobol, I. M. (1967). On the distribution of points in a cube and the approximate evaluation of integrals. *USSR Computational Mathematics and Mathematical Physics* 7(4):86-112. [doi:10.1016/0041-5553%2867%2990144-9](https://doi.org/10.1016/0041-5553%2867%2990144-9).

Sobol sequences.

### Halton (1960) { #halton-1960 }

Halton, J. H. (1960). On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals. *Numerische Mathematik* 2:84-90. [doi:10.1007/BF01386213](https://doi.org/10.1007/BF01386213).

Halton sequences.

### Morris and Mitchell (1995) { #morris-1995 }

Morris, M. D. and Mitchell, T. J. (1995). Exploratory designs for computational experiments. *Journal of Statistical Planning and Inference* 43(3):381-402. [doi:10.1016/0378-3758%2894%2900035-T](https://doi.org/10.1016/0378-3758%2894%2900035-T).

Maximin Latin hypercube designs.

### Sacks et al. (1989) { #sacks-1989 }

Sacks, J., Welch, W. J., Mitchell, T. J. and Wynn, H. P. (1989). Design and analysis of computer experiments. *Statistical Science* 4(4):409-423. [doi:10.1214/ss/1177012413](https://doi.org/10.1214/ss/1177012413).

The foundational paper on designing experiments for deterministic computer models.

### Santner et al. (2018) { #santner-2018 }

Santner, T. J., Williams, B. J. and Notz, W. I. (2018). *The Design and Analysis of Computer Experiments*, 2nd ed. Springer. [doi:10.1007/978-1-4939-8847-1](https://doi.org/10.1007/978-1-4939-8847-1).

The standard textbook on computer experiments and space-filling designs.

## Uncertainty calibration and diagnostics

### Bastos and O'Hagan (2009) { #bastos-2009 }

Bastos, L. S. and O'Hagan, A. (2009). Diagnostics for Gaussian process emulators. *Technometrics* 51(4):425-438. [doi:10.1198/TECH.2009.08019](https://doi.org/10.1198/TECH.2009.08019).

Standardized and pivoted errors, and Q-Q plots for GPs: the closest match to ALchemist's diagnostics.

### Gneiting et al. (2007a) { #gneiting-2007a }

Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *Journal of the Royal Statistical Society B* 69(2):243-268. [doi:10.1111/j.1467-9868.2007.00587.x](https://doi.org/10.1111/j.1467-9868.2007.00587.x).

Calibration and sharpness for probabilistic forecasts.

### Gneiting and Raftery (2007b) { #gneiting-2007b }

Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102(477):359-378. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).

Scoring rules for probabilistic predictions.

### Kuleshov et al. (2018) { #kuleshov-2018 }

Kuleshov, V., Fenner, N. and Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *International Conference on Machine Learning*, PMLR 80:2796-2804. [arXiv:1807.00263](https://arxiv.org/abs/1807.00263).

Recalibration by isotonic regression.

### Guo et al. (2017) { #guo-2017 }

Guo, C. et al. (2017). On calibration of modern neural networks. *International Conference on Machine Learning*, PMLR 70:1321-1330. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).

Temperature scaling, a classification analog of σ-scaling.

### DeGroot and Fienberg (1983) { #degroot-1983 }

DeGroot, M. H. and Fienberg, S. E. (1983). The comparison and evaluation of forecasters. *The Statistician* 32(1/2):12-22. [doi:10.2307/2987588](https://doi.org/10.2307/2987588).

The calibration (reliability) curve.

### Wilk and Gnanadesikan (1968) { #wilk-1968 }

Wilk, M. B. and Gnanadesikan, R. (1968). Probability plotting methods for the analysis of data. *Biometrika* 55(1):1-17. [doi:10.1093/biomet/55.1.1](https://doi.org/10.1093/biomet/55.1.1).

The Q-Q plot.

### Clopper and Pearson (1934) { #clopper-1934 }

Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26(4):404-413. [doi:10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404).

Exact binomial confidence intervals, used for the bands on the calibration curve.

## Split-plot designs

Added with split-plot design support.
