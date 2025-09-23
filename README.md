# DICE: Diffusion Consensus Equilibrium for Sparse-view CT Reconstruction

**Authors:** Leon Suarez-Rodriguez, Roman Jacome, Romario Gualdron-Hurtado, Ana Mantilla-Dulcey, Henry Arguello

**Paper:** Accepted at **CAMSAP 2025**  
**arXiv:** https://arxiv.org/abs/2509.14566

---

## Abstract

Sparse-view computed tomography (CT) reconstruction is fundamentally challenging due to undersampling, leading to an ill-posed inverse problem. Traditional iterative methods incorporate handcrafted or learned priors to regularize the solution but struggle to capture the complex structures present in medical images. In contrast, diffusion models (DMs) have recently emerged as powerful generative priors that can accurately model complex image distributions. In this work, we introduce Diffusion Consensus Equilibrium (DICE), a framework that integrates a two-agent consensus equilibrium into the sampling process of a DM. DICE alternates between: (i) a data-consistency agent, implemented through a proximal operator enforcing measurement consistency, and (ii) a prior agent, realized by a DM performing a clean image estimation at each sampling step. By balancing these two complementary agents iteratively, DICE effectively combines strong generative prior capabilities with measurement consistency. Experimental results show that DICE significantly outperforms state-of-the-art baselines in reconstructing high-quality CT images under uniform and non-uniform sparse-view settings of 15, 30, and 60 views (out of a total of 180), demonstrating both its effectiveness and robustness.

---

## This Repository

This repository contains the code for the paper **"DICE: Diffusion Consensus Equilibrium for Sparse-view CT Reconstruction"** accepted at **CAMSAP 2025**.

---

## Pre-trained Weights

- Google Drive: https://drive.google.com/drive/folders/1gHY-Pp2SDmHXhvLtDdXFpNthbffAeZSs?usp=sharing

---

## Citation

```bibtex
@misc{suarezrodriguez2025dicediffusionconsensusequilibrium,
      title={DICE: Diffusion Consensus Equilibrium for Sparse-view CT Reconstruction}, 
      author={Leon Suarez-Rodriguez and Roman Jacome and Romario Gualdron-Hurtado and Ana Mantilla-Dulcey and Henry Arguello},
      year={2025},
      eprint={2509.14566},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.14566}, 
}
