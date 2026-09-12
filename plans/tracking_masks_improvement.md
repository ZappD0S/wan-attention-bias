The issue you are encountering is a classic problem when adapting attention mechanisms for tracking in diffusion models. 

During the early denoising steps, the latent representation $X$ is dominated by Gaussian noise. Consequently, the queries ($Q$) and keys ($K$) contain very little semantic information. When you compute the softmax over $Q K^\top$, random noise spikes are amplified, causing tokens in the target frame to arbitrarily attend to random locations in the first frame. Because your binarization step relies on a strict $\arg\max$, even a microscopic advantage caused by noise will cause a pixel to flip from "background" to an "entity," resulting in the flickering/fluctuating regions you see.

To fix this and achieve smooth, uniform tracking masks, you can implement one or a combination of the following strategies. I have ordered them from the easiest to implement to the most robust mathematically.

### 1. Introduce a Confidence Threshold (Soft $\arg\max$)
Right now, you assign a pixel to an entity if it simply has the highest probability, even if that probability is $0.05$ and the background is $0.04$. You should force a "confidence margin" or threshold, defaulting to the background class if no entity strongly claims the pixel.

Modify your binarization equation from:
$$ \hat{M}_{f,i}^{(c)} = \mathbb{I} \left( c = \mathop{\arg\max}_{c' \in \mathcal{C}} M_{f,i}^{(c')} \right) $$

To a thresholded version. If the background class is $c_{bg}$:
$$ 
\hat{M}_{f,i}^{(c)} = 
\begin{cases} 
1 & \text{if } c = \underset{c' \in \mathcal{C} \setminus \{c_{bg}\}}{\operatorname*{argmax}} M_{f,i}^{(c')} \text{ AND } M_{f,i}^{(c)} > \tau \\
1 & \text{if } c = c_{bg} \text{ AND all other classes } \le \tau \\
0 & \text{otherwise}
\end{cases}
$$
*Recommendation:* Start with a threshold $\tau$ between $0.2$ and $0.4$.

### 2. Gaussian Spatial Smoothing (Before Binarization)
Because the attention maps are inherently noisy in early steps, the continuous masks $M_f^{(c)}$ will look like salt-and-pepper noise. Applying a 2D Gaussian blur to the continuous mask *before* the $\arg\max$ operation will average out the noise spikes while preserving the larger, contiguous blob of the actual entity.

1. Reshape the flat continuous mask $M_f^{(c)} \in \mathbb{R}^{HW}$ back to 2D $\mathbb{R}^{H \times W}$.
2. Apply a Gaussian filter $G_\sigma$: 
   $$ \tilde{M}_f^{(c)} = G_\sigma * M_f^{(c)} $$
3. Flatten back to $\mathbb{R}^{HW}$ and apply your $\arg\max$ to $\tilde{M}_f^{(c)}$.

### 3. Inject a Spatial Locality Prior (Distance Penalty)
Entities in video rarely teleport across the screen between frame 1 and frame $f$. You can enforce this physical prior by penalizing attention scores between spatial locations that are far apart. This forces the model to only look near the entity's original location.

Create a static distance matrix $D \in \mathbb{R}^{(HW) \times (HW)}$ where $D_{i,j}$ is the squared Euclidean distance between spatial location $i$ in frame $f$ and location $j$ in frame 1. 

Modify the attention computation to subtract this distance penalty before the softmax:
$$ A_f = \frac{1}{h} \sum_{l=1}^{h} \text{softmax}\left(\frac{Q_{f,l} K_{1,l}^\top}{\sqrt{d}} - \lambda D\right) $$
*Why this works:* Even if the pure noise creates a high dot-product score between a background pixel and the character, the distance penalty $(-\lambda D)$ will push the pre-softmax logit to a highly negative value, reducing its attention weight to zero.

### 4. Timestep-Dependent Mask Relaxation
Since the noise is exclusively a problem in the early denoising steps (high $t$), you can bypass or dampen the tracking mechanism when $t$ is large.

**Option A (Static early steps):** For the first 20-30% of the diffusion steps, simply copy the first frame mask to all frames without computing $A_f$.
$$ M_f^{(c)} = M_1^{(c)} \quad \text{for } t > t_{threshold} $$
*Rationale:* Early steps dictate global layout. The entity hasn't moved yet in the latent representation.

**Option B (Linear Interpolation):** Blend the tracking mask with the static first-frame mask based on the timestep $t$ (where $t=T$ is pure noise, $t=0$ is clear video). Let $\alpha_t = t/T$:
$$ \tilde{M}_f^{(c)} = \alpha_t M_1^{(c)} + (1 - \alpha_t) \left( M_1^{(c)} A_f^\top \right) $$

### 5. Aggregate Attention Across Layers
If you are currently extracting $Q$ and $K$ from a single DiT block, it might be heavily affected by the specific feature representation of that block. 
* Early layers often contain pure spatial high-frequency noise.
* Middle-to-deep layers contain better semantic correspondence.

Instead of one block, extract $Q$ and $K$ from 2 or 3 middle blocks of the Wan2.1 transformer and average the resulting $A_f$ matrices. This dramatically stabilizes the mask generation.

### Summary of what to implement first:
If you want the quickest fix that requires the least code changes, implement **Gaussian Smoothing** (applied to the $H \times W$ mask before binarization) and the **Confidence Threshold**. 

If you are looking for the most mathematically elegant solution to put into your paper draft, adding the **Spatial Locality Prior (Distance Penalty)** directly addresses the underlying cause (global attention on noisy latents) by introducing an inductive bias suited for video motion.
