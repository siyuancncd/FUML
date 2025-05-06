# FUML

Siyuan Duan, Yuan Sun, Dezhong Peng, Guiduo Duan, Xi Peng, and Peng Hu. "Deep Fuzzy Multi-view Learning for Reliable Classification". (ICML 2025, PyTorch Code)

## 

:bangbang: **I’m actively seeking a PhD position for Fall 2026 entry.** If you believe my background aligns with your research needs, please feel free to contact me via email at siyuanduancn@gmail.com.

## Abstract

Multi-view learning methods primarily focus on enhancing decision accuracy but often neglect the uncertainty arising from the intrinsic drawbacks of data, such as noise, conflicts, etc. To address this issue, several trusted multi-view learning approaches based on the Evidential Theory have been proposed to capture uncertainty in multi-view data. However, their performance is highly sensitive to conflicting views, and their uncertainty estimates, which depend on the total evidence and the number of categories, often underestimate uncertainty for conflicting multi-view instances due to the neglect of inherent conflicts between belief masses. To accurately classify conflicting multi-view instances and precisely estimate their intrinsic uncertainty, we present a novel Deep Fuzzy Multi-View Learning (**FUML**) method. Specifically, FUML leverages Fuzzy Set Theory to model the outputs of a classification neural network as fuzzy memberships, incorporating both possibility and necessity measures to quantify category credibility. A tailored loss function is then proposed to optimize the category credibility. To further enhance uncertainty estimation, we propose an entropy-based uncertainty estimation method leveraging category credibility. Additionally, we develop a Dual Reliable Multi-view Fusion (DRF) strategy that accounts for both view-specific uncertainty and inter-view conflict to mitigate the influence of conflicting views in multi-view fusion. Extensive experiments demonstrate that our FUML achieves state-of-the-art performance in terms of both accuracy and reliability. 

## Motivation

<p align="center">
<img src="https://github.com/siyuancncd/FUML/blob/main/FUML_motivations.png" width="440" height="360">
</p>

(a) Visualization of the conflicting multi-view instance: the depth view is related to the ``Bedroom'' category, while the other views show conflicting information, such as ``Bathroom.'' (b) EDL-based TMVC methods are sensitive to such conflicting multi-view instances. On one hand, because they neglect the global conflict between views in multi-view fusion, classification errors are often made. On the other hand, their uncertainty estimation is only related to the total evidence and the number of categories. For conflicting multi-view instances, as long as the total evidence is large, the uncertainty is seriously underestimated. (c) In our method, both global conflict and uncertainty are considered during fusion, allowing the conflicting multi-view instances to be classified correctly. Additionally, this method can estimate decision uncertainty more accurately.

## Framework

## Experiments

## Requirements

## Datasets

## Train and test

The code is coming soon...

## Citation

coming soon...

## Question?

If you have any questions, please email ddzz12277315 AT 163 DOT com or siyuanduancn AT gmail DOT com.

## Acknowledgement

