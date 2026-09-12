# Concept Weaver: Enabling Multi-Concept Fusion in Text-to-Image Models

**Gihyun Kwon$^1$, Simon Jenni$^2$, Dingzeyu Li$^2$, Joon-Young Lee$^2$, Jong Chul Ye$^1$, Fabian Caba Heilbron$^2$**
$^1$KAIST, $^2$Adobe
`[gihyun, jong.ye]@kaist.ac.kr`, `[jenni, dinli, jolee, caba]@adobe.com`

---

## Abstract
While there has been significant progress in customizing text-to-image generation models, generating images that combine multiple personalized concepts remains challenging. In this work, we introduce Concept Weaver, a method for composing customized text-to-image diffusion models at inference time. Specifically, the method breaks the process into two steps: creating a template image aligned with the semantics of input prompts, and then personalizing the template using a concept fusion strategy. The fusion strategy incorporates the appearance of the target concepts into the template image while retaining its structural details. The results indicate that our method can generate multiple custom concepts with higher identity fidelity compared to alternative approaches. Furthermore, the method is shown to seamlessly handle more than two concepts and closely follow the semantic meaning of the input prompt without blending appearances across different subjects.

---

## 1. Introduction
Text-to-image generation models have shown impressive capabilities [21, 23, 28] in the last few years. Existing open source [21] and commercial solutions such as Adobe Firefly have enabled aspiring creatives to generate images with unprecedented quality by simply crafting text prompts. Progress has also been attained in developing models that can customize images for your own subjects or visual concepts [3, 11, 22, 25]. These technologies have opened the door for new ways of content creation, where aspiring creators can craft stories with personalized characters under different scenes and styles.

While there has been significant progress in customizing text-to-image generation models, generating images that combine multiple personalized concepts remains challenging. Several approaches [11, 25] offer the ability to jointly train models for multiple concepts or merge customized models, enabling the creation of scenes with more than one personalized concept. However, it often fails to generate semantically related concepts (e.g., cat and dog) and struggles to scale beyond three or more concepts. More recently, Mix-of-show [4] has addressed the issue of multi-concept generation with disentangled Low-Rank (LoRa) [9] weight merging and regional guidance at the sampling stage. However, the model still suffers from mixed concepts due to the difficulty of weight merging.

In this paper, we propose a tuning-free method for composing customized text-to-image diffusion models at inference time. We illustrate our key idea in **Figure 2**, where the goal is to generate images featuring more than two custom concepts. Specifically, rather than generating a personalized image from scratch, we break the process into two steps: first, we create a template image that aligns with the semantics of the input prompt, and then we personalize this template image using a novel concept fusion strategy. The fusion strategy takes as input the non-personalized template image along with region concept guidance (obtained automatically) to generate an edited image that retains the template’s structural details while incorporating the target concepts’ appearance and style. This fusion approach injects concept details into specific spatial regions, allowing us to compose multiple concepts (from the Bank of Concepts) in generated images without blending appearances across different subjects.

---

## 3. The Concept Weaver’s Method
Concept Weaver employs a cascading generation process. Consider the prompt: *"A [C1]dog and a [C2]cat playing with a ball, [C3]mountain background"*, where $[C1, C2, C3]$ denote custom concepts. Our approach begins by personalizing text-to-image models for each concept (**Step 1**). Next, we select a non-personalized ‘template image’ using the given prompt (**Step 2**). In the third step, we extract latent representations from this template to aid in later editing (**Step 3**). The fourth step involves identifying and isolating the specific regions of the template image that correspond to the target subjects (**Step 4**). Finally, our key contribution (**Step 6**) combines these latent representations, targeted spatial regions, and personalized models to reconstruct the template image, infusing it with the specified concepts.

### Step 1: Concept Bank Training
We fine-tune a pretrained text-to-image model for each concept. We leverage Custom Diffusion [11], fine-tuning only the ‘key’ and the ‘value’ weight parameters $W^k, W^v$ of the cross-attention layers. With text condition $p \in \mathbb{R}^{s \times d}$ and self-attention feature $f \in \mathbb{R}^{(h \times w) \times c}$, the cross-attention layer consists of:
$$Q = W^q f, K = W^k p, V = W^v p$$

### Step 2: Template Image Generation
We start from a template image that can be customized. These images should include the semantic objects with the background desired in the prompt. We generate template images using Stable Diffusion [21] version $\geq$ 2.0.

### Step 3: Inversion and Feature Extraction
We apply an inversion process to obtain a latent representation. We borrow the image inversion and feature extraction schemes from plug-and-play diffusion (PNP) [26]. From source image $x_{src}$, we generate noisy latent $z_T$ with the DDIM [24] forward process. During the reverse reconstruction, we extract features $f_t^l$ from the U-Net’s $l$-th layer at each timestep $t$.

### Step 4: Mask Generation
To obtain semantic mask regions, we leverage the Segment Anything Model (SAM) [10] and the pre-trained text conditional grounding model [15]. For $N$ concepts, we extract concept-wise masks $M_1, M_2, \dots, M_N$ and set the unmasked region as background mask $M_{bg} = (M_1 \cup M_2 \cup \dots \cup M_N)^c$.

### Step 5: Multi-Concept Fusion
We combine multiple single-concept personalized models in a unified sampling process. A naive approach to mix multiple score estimations is:
$$\epsilon_{fuse} = \sum_{i}^{N} \epsilon_{\theta_i}(z_t, t, p_{+i})M_i + \epsilon_{\theta_{bg}}(z_t, t, p_{+bg})M_{bg}$$
where $p_{+i}$ are concept-aware prompts. We further improve this by mixing concepts in the feature space of cross-attention layers:
$$h_{fuse} = \sum_{i}^{N} h_i M_i + h_{bg} M_{bg}$$
To remove concept-free features, we propose a suppression method:
$$h_{fuse} = (1 + \lambda)\left[\sum_{i}^{N} h_i M_i + h_{bg} M_{bg}\right] - \lambda h_{base}$$
Finally, the fused score estimation is:
$$\epsilon_{fuse} = \epsilon_\theta(z_t, t; h_{fuse}; f_t)$$

---

## 4. Experimental Results
We evaluate our multi-concept fusion approach against baselines like Custom Diffusion [11], Textual Inversion [3], Perfusion [25], and Mix-of-show [4].

**Table 1: Quantitative Evaluation of Multi-Concept Generation**
| Method | Text sim $\uparrow$ | Image sim $\uparrow$ |
| :--- | :---: | :---: |
| Textual Inversion | 0.3423 | 0.7256 |
| Custom Diffusion | 0.3595 | 0.7875 |
| Perfusion | 0.3182 | 0.7563 |
| Mix-of-show | 0.3634 | 0.7984 |
| **Concept Weaver (ours)** | **0.3804** | **0.8124** |

**Table 2: Human Preference Study**
| Method | Text match $\uparrow$ | Concept match $\uparrow$ | Realism $\uparrow$ |
| :--- | :---: | :---: | :---: |
| Mix-of-show | 3.44 | 3.39 | 3.78 |
| **Concept Weaver (Ours)** | **4.70** | **4.64** | **4.43** |

---

## 5. Conclusion
We introduced a novel framework to generate high-fidelity images containing multiple custom concepts. Our method fuses multiple personalized models during the sampling stage without additional optimization. Experimental results show our method outperforms state-of-the-art customization methods in multiple axes.
