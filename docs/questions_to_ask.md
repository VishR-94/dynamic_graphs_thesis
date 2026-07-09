1) Forecasting task - should we use windowed features? This would mean each sample is of shape [context_length, num_assets, num_channels] and we will be trying to map this onto [forecasting_horizon, num_assets, tgt_channels].
2) Comparison metrics - some models arent probabilistic so cant be compared to a likelihood. Is it OK to use something like MSE or MAP as a benchmark metric to compare all models?
3) How do we do the probabilistic output head for our architecture - we output discrete tokens. Should we (a) output a softmax prob vector over the discrete token space and then sample from that (before feeding into the decoder), giving us essentially an emperical distribution over the predictions in the continuous space or (b) in our forecasting model take the forecasted discrete tokens, decode back to the continuous space and then feed that into a probabilistic output head that gives us the mean and variance of a Gaussian as our output?
4) Should we standardise our data beforehand? If so, how? Using the mean and std of the training dataset or should we standardise each window individually?

5) It looks like theres some errors in the data - examples found:
    1. NVDA prices look incorrect - first half of the year the level is completely wrong
    2. BKNG prices look incorrect - shape looks fine but the level is wrong. Data shows it trading around 3500 but it traded 150 in 2024
    3. AVGO prices look incorrect - first half of the year the level is completely wrong
    4. WMT prices look incorrect - first few months of year the level is wrong

------
1) Ask Lampos - when we evaluate the statistical benchmark models (ARIMA,GARCH), is it ok if we skip steps since its not feasible to evaluate on exactly the same steps as the neural models since the statistical models will be univariate, 1 per channel per asset. This means on the test dataset we will need to do 5,691 × 93 × 5 = 2.65 million forecasts which wont be feasible. Instead can we just stride forward our context window (so we jump maybe 60minutes ahead) which reduces the number of forecasts to 48,825. Is that defensible just for those benchmarks (not an issue for the neural models since they are multivariate and can be batched).
    
