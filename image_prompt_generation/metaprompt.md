**Role:** You are an expert dataset generator for AI video synthesis.
**Task:** Generate a JSON file containing a `safeguard_suffix` and a `dataset` of 10 distinct scenes.
**Subject Matter:** Scenes depicting two similar or identical characters (and optional inanimate objects) against simple/neutral backgrounds.

#### 1. Universal Safeguards (Global Constant)
For the `safeguard_suffix` key, you must output **exactly** this string. Do not alter it:
> **"The scene is filmed as a continuous shot with a static camera, ensuring no cuts, no new objects entering, and no limbs merging, while maintaining constant lighting and character consistency."**

#### 2. Scene & Character Constraints
*   **Characters:** Exactly two characters per scene. Similar or identical.
*   **Background:** Simple, neutral, or out of focus.
*   **Independence:** Characters must never interact with each other physically.
*   **Movement:** Actions must be dynamic but stationary (e.g., eating, typing).
*   **State Physics & Causality:**
    *   **Universal Compatibility:** **Action X** and **Action Y** must share a compatible physical starting state.
    *   **Identical Initial State:** The `appearance_prompt` must describe **both characters exactly the same way**.
    *   **Visual Evidence & Preconditions:** **Do not rely on implied mechanisms.** Actions must be physically grounded in the initial appearance.
        *   If an action involves **emitting** something (e.g., light, water, smoke, sound), the **emitter** (e.g., a raised flash unit, a visible nozzle, a speaker cone) must be explicitly described as present and exposed in the `appearance_prompt`.
        *   If an action involves **manipulating** a part (e.g., typing, biting), the **limb or feature** (e.g., hands on keyboard, visible teeth) must be positioned to allow it.
        *   **Rule of Thumb:** If the tool required for the action is hidden or closed in the `appearance_prompt`, the action is invalid.

#### 3. The Prompt Types
For each scene, generate the `appearance_prompt` and 3 variations of action prompts. **Plan Actions X and Y first**, then reverse-engineer the appearance to include every tool, limb, or feature necessary to perform those actions.

1.  **`appearance_prompt`** (Scene Setup): Visual state *before* movement.
2.  **`default`** (Action Prompt 1): A performs **Action X**. B performs **Action Y** (Standard locative description).
3.  **`no_locative`** (Action Prompt 2):
    *   **Structure:** Single sentence connecting Action X and Action Y with "and".
    *   **Constraints:** Do not use spatial words (e.g., "left", "right", "center", "middle", "side", "background").
4.  **`split_sentences`** (Action Prompt 3):
    *   **Structure:** Two grammatically complete, independent sentences. One for Action X, one for Action Y.
    *   **Constraints:**
        *   **No Spatial Words:** Do not use spatial words (e.g., "left", "right", "center", "middle").
        *   **No Distinctions:** Do not use words that distinguish the characters (e.g., "another", "the other", "the second"). You must refer to the subject exactly the same way in both sentences.
    *   **Example:** "The dog barks. The dog sticks its tongue out."

#### 4. Segmentation & Masking Rules
Every prompt object must contain a `segments` array and a `mask` array.
*   **Nested Structure:** Both `segments` and `mask` must be **arrays of arrays**.
    *   Each inner array represents one sentence.
    *   **Single-Sentence Prompts (`appearance`, `default`, `no_locative`):** The outer array contains exactly one inner array.
    *   **Two-Sentence Prompts (`split_sentences`):** The outer array contains exactly two inner arrays.
*   **Mask Values:**
    *   **`1` (Character Segment):** Must contain the **Subject** + **State/Action** + **Relevant Objects**.
    *   **`0` (Context Segment):** Contains locative expressions ("On the left,"), connectors ("and"), or background descriptions. Do *not* include the safeguard text.
*   **Mask Count:** There must be exactly **two** segments marked with `1` across the entire prompt structure.
    *   *For single-sentence prompts:* The single inner array contains two `1`s.
    *   *For split-sentence prompts:* Each of the two inner arrays contains exactly one `1`.
*   **Formatting:** The last segment of every inner array (sentence) must end with a period.

#### 5. Output Format
Return **only** valid JSON matching this structure exactly.

```json
{
  "safeguard_suffix": "The scene is filmed as a continuous shot with a static camera, ensuring no cuts, no new objects entering, and no limbs merging, while maintaining constant lighting and character consistency.",
  "dataset": [
    {
      "appearance_prompt": {
        "segments": [
          ["On the left,", "a student sits with a closed laptop", "and on the right,", "a student sits with a closed laptop", "at a wooden desk."]
        ],
        "mask": [
          [0, 1, 0, 1, 0]
        ]
      },
      "action_prompts": {
        "default": {
          "segments": [
            ["On the left,", "the student opens the laptop to begin typing", "while on the right,", "the student picks up the closed laptop to put it in a bag."]
          ],
          "mask": [
            [0, 1, 0, 1]
          ]
        },
        "no_locative": {
          "segments": [
            ["A student opens a laptop to begin typing", "and", "a student picks up a closed laptop to put it in a bag."]
          ],
          "mask": [
            [1, 0, 1]
          ]
        },
        "split_sentences": {
          "segments": [
            ["A student opens a laptop to begin typing."],
            ["A student picks up a closed laptop to put it in a bag."]
          ],
          "mask": [
            [1],
            [1]
          ]
        }
      }
    }
    // ... Repeat for 10 items
  ]
}
```
