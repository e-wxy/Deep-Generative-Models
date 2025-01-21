# Deep Generative Models for Style Transfer

Exploring state-of-the-art deep generative models (DGMs) for creating Impressionist-style paintings.


<img src="Assets/img/Diffused_Aalto.png" alt="Aalto campus" width="800">


## Installation

### Environment
Create conda environment:

```shell
conda env create -f environment.yml
```

### Model

Download the pretrained model weights and place them in the `MODEL_NAME/checkpoints` directory.
e.g.
```shell
cd Stable-Diffusion/checkpoints
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

#### Stable Diffusion

**SDXL-1.0**:

- [base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/)
- [refiner model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/)


#### Guided Diffusion

- [diffusion](https://aaltofi-my.sharepoint.com/:f:/g/personal/xinyi_wen_aalto_fi/EnGSIn8ytKBNoJxAsb578vwB1XvhWvrfUcA8GrET4maezQ?e=CoKoBv)

## Run
Open Jupyter Notebook and run all cells:

- [SDXL](Stable-Diffusion/SDXL.ipynb)
- [Guided Diffusion](CLIP-Guided/Guided-Diffusion.ipynb)
- [Neural Style Transfer](NST/NST.ipynb)
- [CycleGAN](CycleGAN/CycleGAN.ipynb)

Config params in `Run->Test`.

## References
1. Podell, Dustin, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. ‘SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis’. In The Twelfth International Conference on Learning Representations, 2023. https://openreview.net/forum?id=di52zR8xgf.
2. Dhariwal, Prafulla, and Alexander Nichol. ‘Diffusion Models Beat GANs on Image Synthesis’. In Advances in Neural Information Processing Systems, 34:8780–94. Curran Associates, Inc., 2021. https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html.
3. Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. ‘Image Style Transfer Using Convolutional Neural Networks’. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2414–23. Las Vegas, NV, USA: IEEE, 2016. https://doi.org/10.1109/CVPR.2016.265.
4. Zhu, Jun-Yan, Taesung Park, Phillip Isola, and Alexei A. Efros. ‘Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks’. In 2017 IEEE International Conference on Computer Vision (ICCV), 2242–51. Venice: IEEE, 2017. https://doi.org/10.1109/ICCV.2017.244.


## Acknowledgement
Codebase from [Generative Models by Stability AI](https://github.com/Stability-AI/generative-models), [Guided Diffusion](https://github.com/openai/guided-diffusion).