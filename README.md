# FUML

Siyuan Duan, Yuan Sun, Dezhong Peng, Guiduo Duan, Xi Peng, and Peng Hu. "[Deep Fuzzy Multi-view Learning for Reliable Classification](https://openreview.net/forum?id=ZzuaeYvLsJ)". (ICML 2025, PyTorch Code)

- Paper is available at [here](https://github.com/siyuancncd/FUML/blob/main/src/ICML25_Deep_Fuzzy_Multi_view_Learning_for_Reliable_Classification.pdf).
- Poster is available at [here](https://github.com/siyuancncd/FUML/blob/main/FUML_poster.png).
- PPT is available at [here](https://github.com/siyuancncd/FUML/blob/main/FUML-AITIME-Intro.pdf).
- 中文讲解 在 [这里](https://www.bilibili.com/video/BV1mA3yzEEWF/?spm_id_from=333.1387.homepage.video_card.click&vd_source=6c3619d976449503d4bb811d9044ee7f&t=5481).

## Abstract

Multi-view learning methods primarily focus on enhancing decision accuracy but often neglect the uncertainty arising from the intrinsic drawbacks of data, such as noise, conflicts, etc. To address this issue, several trusted multi-view learning approaches based on the Evidential Theory have been proposed to capture uncertainty in multi-view data. However, their performance is highly sensitive to conflicting views, and their uncertainty estimates, which depend on the total evidence and the number of categories, often underestimate uncertainty for conflicting multi-view instances due to the neglect of inherent conflicts between belief masses. To accurately classify conflicting multi-view instances and precisely estimate their intrinsic uncertainty, we present a novel Deep Fuzzy Multi-View Learning (**FUML**) method. Specifically, FUML leverages Fuzzy Set Theory to model the outputs of a classification neural network as fuzzy memberships, incorporating both possibility and necessity measures to quantify category credibility. A tailored loss function is then proposed to optimize the category credibility. To further enhance uncertainty estimation, we propose an entropy-based uncertainty estimation method leveraging category credibility. Additionally, we develop a Dual Reliable Multi-view Fusion (DRF) strategy that accounts for both view-specific uncertainty and inter-view conflict to mitigate the influence of conflicting views in multi-view fusion. Extensive experiments demonstrate that our FUML achieves state-of-the-art performance in terms of both accuracy and reliability. 

## Motivation

<p align="center">
<img src="https://github.com/siyuancncd/FUML/blob/main/FUML_motivations.png" width="440" height="360">
</p>

(a) Visualization of the conflicting multi-view instance: the depth view is related to the ``Bedroom'' category, while the other views show conflicting information, such as 'Bathroom'. (b) EDL-based TMVC methods are sensitive to such conflicting multi-view instances. On one hand, because they neglect the global conflict between views in multi-view fusion, classification errors are often made. On the other hand, their uncertainty estimation is only related to the total evidence and the number of categories. For conflicting multi-view instances, as long as the total evidence is large, the uncertainty is seriously underestimated. (c) In our method, both global conflict and uncertainty are considered during fusion, allowing the conflicting multi-view instances to be classified correctly. Additionally, this method can estimate decision uncertainty more accurately.

## Framework

<p align="center">
<img src="https://github.com/siyuancncd/FUML/blob/main/FUML_framework.png" width="1000" height="380">
</p>

## Experimental Results

<p align="center">
<img src="https://github.com/siyuancncd/FUML/blob/main/FUML_results1.png" width="1000" height="400">
</p>

<p align="center">
<img src="https://github.com/siyuancncd/FUML/blob/main/FUML_results2.png" width="1000" height="200">
</p>

## Get Started!

1️⃣ Install required libraries:

### Requirements

```
Python==3.9.0
torch==2.3.1
torchvision==0.18.1
numpy==1.26.4
scikit-learn==1.5.0
scipy==1.13.1
```

2️⃣ Download the datasets and put them in the **datasets** folder.

### Datasets

All datasets can be downloaded from 

[https://github.com/YilinZhang107/Multi-view-Datasets](https://github.com/YilinZhang107/Multi-view-Datasets)

[https://gitee.com/zhangfk/multi-view-dataset/tree/master](https://gitee.com/zhangfk/multi-view-dataset/tree/master)

[https://github.com/JethroJames/Awesome-Multi-View-Learning-Datasets](https://github.com/JethroJames/Awesome-Multi-View-Learning-Datasets).

3️⃣ Run the code.

### Train and test

```
python main.py
```

## Citation
If this codebase is useful for your work, please cite our papers:
```
@inproceedings{duandeep,
  title={Deep Fuzzy Multi-view Learning for Reliable Classification},
  author={Duan, Siyuan and Sun, Yuan and Peng, Dezhong and Duan, Guiduo and Peng, Xi and Hu, Peng},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

```
@inproceedings{duan2025fuzzy,
  title={Fuzzy Multimodal Learning for Trusted Cross-modal Retrieval},
  author={Duan, Siyuan and Sun, Yuan and Peng, Dezhong and Liu, Zheng and Song, Xiaomin and Hu, Peng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20747--20756},
  year={2025}
}
```
## Question?

If you have any questions, please email siyuanduancn AT gmail DOT com.

## Acknowledgement

The code is inspired by [Fuzzy Multimodal Learning for Trusted Cross-modal Retrieval](https://github.com/siyuancncd/FUME) (CVPR 2025).
