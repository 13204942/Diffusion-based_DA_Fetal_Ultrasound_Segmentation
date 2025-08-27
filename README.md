# Diffusion Model-based Data Augmentation Method for Fetal Head Ultrasound Segmentation
Source codes of the conference paper ([DOI: 10.48550/arXiv.2506.23664](https://doi.org/10.48550/arXiv.2506.23664)) accepted at **Irish Machine Vision and Image Processing Conference (IMVIP) 2025** (<span style="color:blue">Oral</span>).

![Architecture](assests/ft_sam.drawio.png)

## Data

### Task: Diffusion Fine-tuning
| Source    | \# Images | Purpose |
| -------- | ------- | ------- |
| HC18  | 20 $\times$ 3  | Fine-tuning |

### Task: SAM Fine-tuning
| Source    | \# Images | Purpose |
| -------- | ------- | ------- |
| HC18  |   500  | Fine-tuning |
| Spain (ES) |   500   | Fine-tuning |
| Synthesis |   495   | Fine-tuning |

### Task: Domain Generalization Evaluation
| Source    | \# Images | Purpose |
| -------- | ------- | ------- |
| Spain (ES) |   597   | Testing |
| African (AF)    |   125  | Testing |

## Results

![Synthetic Images](assests/synthetic_img.drawio.png)

![FT Results](assests/linechart.drawio.png)
