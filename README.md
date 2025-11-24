# RobotSeg

RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video

# 🤖 RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video

RobotSeg is a dedicated foundation model and large-scale dataset for **robot segmentation in both images and videos**.  
It is built upon **SAM 2** and introduces architecture enhancements that make segmentation of robot embodiments more accurate, robust, and structure-preserving.

**Authors:** Haiyang Mei, Qiming Huang, Hai Ci, Mike Zheng Shou  
**Affiliation:** Show Lab, National University of Singapore  

📄 **Paper PDF:** RobotSeg.pdf

---

## 🚀 Highlights

### ⭐ RobotSeg Model
- Extends **SAM 2** into a robot-aware video segmentation model  
- Supports **zero-click automatic segmentation**  
- Produces **structure-preserving masks**, avoiding broken links or missing grippers  
- Compact: **41.3M** parameters, smaller than robot-specific and language-conditioned baselines

### ⭐ VRS Dataset (Video Robot Segmentation)
- **2,812 videos**, **138,707 labeled frames**  
- 10+ robot embodiments  
- Hierarchical labels: robot **arm**, **gripper**, **whole robot**  
- Training uses **only first-frame annotations**, enabling label-efficient learning

### ⭐ Superior Performance
RobotSeg outperforms:
- Robot-specific models  
- Language-conditioned models  
- SAM 2.1 baseline

Across:
- Automatic (0-click)
- 1-click / 3-click prompts
- Bounding-box prompts
- Online interactive refinement

---

## 🧩 Method Overview

### 1. Structure-Enhanced Memory Associator (SEMA)
- Injects multi-scale structure cues  
- Preserves thin robot parts  
- Improves temporal coherence  

### 2. Robot Prompt Generator (RPG)
- Provides robot-aware prompt tokens  
- Enables **automatic robot segmentation**  

### 3. Label-Efficient Training Strategy (LET)
- Cycle consistency  
- Semantic consistency  
- Patch-level consistency  

---

## 🎥 VRS Dataset

- Long video sequences  
- Diverse scenes  
- Fine-grained hierarchical labels  
- Fully annotated test set  
- First-frame annotated training set  

---

## 📦 Getting Started (Template)

```bash
git clone https://github.com/yourname/RobotSeg.git
cd RobotSeg

conda create -n robotseg python=3.10
conda activate robotseg
pip install -r requirements.txt
```

---

## 🔖 BibTeX

```bibtex
@misc{mei2025robotseg,
  title        = {RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video},
  author       = {Haiyang Mei and Qiming Huang and Hai Ci and Mike Zheng Shou},
  year         = {2025},
  note         = {Preprint},
}
```

---

## 🙌 Acknowledgments

RobotSeg is developed at **Show Lab, National University of Singapore**.
