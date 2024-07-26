# Prior-EDL
## Simulated SAR Prior Knowledge guided Evidential Deep Learning for Reliable Few-Shot SAR Target Recognition

## Abstract
Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) plays a pivotal role in civilian and military applications. However, the limited labeled samples present a significant challenge in deep learning-based SAR ATR. Few-shot learning (FSL) offers a potential solution, but models trained with limited samples may produce a high probability of incorrect results that can mislead decision-makers. To address this, we introduce uncertainty estimation into SAR ATR and propose Prior knowledge-guided Evidential Deep Learning (Prior-EDL) to ensure reliable recognition in FSL. Inspired by Bayesian principles, Prior-EDL leverages prior knowledge for improved predictions and uncertainty estimation. We use a deep learning model pre-trained on simulated SAR data to discover category correlations and represent them as label distributions. This knowledge is then embedded into the target model via a Prior-EDL loss function, which selectively uses the prior knowledge of samples due to the distribution shift between simulated data and real data. To unify the discovery and embedding of prior knowledge, we propose a framework based on the teacher-student network. Our approach enhances the model's evidence assignment, improving its uncertainty estimation performance and target recognition accuracy. Extensive experiments on the MSTAR dataset demonstrate the effectiveness of Prior-EDL, achieving recognition accuracies of 70.19\% and 92.97\% in 4-way 1-shot and 4-way 20-shot scenarios, respectively. For Out-Of-Distribution data, Prior-EDL outperforms other uncertainty estimation methods. The code is available at https://github.com/Xiaoyan-Zhou/Prior-EDL/.

## Usage
### train

```sh
python 
```

### test

```sh
python 
```


