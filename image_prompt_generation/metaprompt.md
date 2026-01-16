**Role:** You are an expert dataset generator for AI video synthesis.
**Task:** Generate a JSON file containing a `safeguard_suffix` and a `dataset` of 10 distinct scenes.
**Subject Matter:** Scenes depicting two similar or identical characters (and optional inanimate objects) against simple/neutral backgrounds.

#### 1. Universal Safeguards (Global Constant)
For the `safeguard_suffix` key, you must output **exactly** this string. Do not alter it:
> **"The scene is filmed as a continuous shot with a static camera, ensuring no cuts and no new objects entering."**

#### 2. Scene & Character Constraints
*   **Characters:** Exactly two characters per scene. Similar or identical.
*   **Background:** Simple, neutral, or out of focus.
*   **Independence:** Characters must never interact with each other physically.
*   **Movement:** Actions must be dynamic but stationary (e.g., eating, typing, waving).
*   **State Physics & Causality:**
    *   **Universal Compatibility:** **Action X** and **Action Y** must share a compatible physical starting state.
    *   **Identical Initial State:** The `appearance_prompt` must describe **both characters exactly the same way**.
    *   **Visual Evidence & Preconditions:** **Do not rely on implied mechanisms.** Actions must be physically grounded in the initial appearance.
        *   If an action involves **emitting** something (e.g., light, water, smoke), the **emitter** (e.g., flash unit, nozzle) must be explicitly described in the `appearance_prompt`.
        *   If an action involves **manipulating** a tool, that tool must be present in the `appearance_prompt`.

#### 3. The Prompt Types
For each scene, generate the `appearance_prompt` and 3 variations of action prompts. **Plan Actions X and Y first**, then reverse-engineer the appearance.

1.  **`appearance_prompt`** (Scene Setup): Visual state *before* movement.
2.  **`default`** (Action Prompt 1): A performs **Action X**. B performs **Action Y** (Standard locative description).
3.  **`no_locative`** (Action Prompt 2):
    *   **Structure:** Single sentence connecting Action X and Action Y with "and".
    *   **Constraints:** Do not use spatial words (e.g., "left", "right", "center", "middle", "side", "background").
4.  **`split_sentences`** (Action Prompt 3):
    *   **Structure:** Two grammatically complete, independent sentences. One for Action X, one for Action Y.
    *   **Constraints:**
        *   **No Spatial Words:** Do not use spatial words.
        *   **No Distinctions:** Do not use words that distinguish the characters (e.g., "another", "the second"). You must refer to the subject exactly the same way in both sentences.
    *   **Example:** "The dog barks. The dog sticks its tongue out."

#### 4. Segmentation & Masking Rules
Every prompt object must contain a `segments` array and a `mask` array.
*   **Nested Structure:** Both `segments` and `mask` must be **arrays of arrays**.
    *   Each inner array represents one sentence.
    *   **Single-Sentence Prompts:** The outer array contains exactly one inner array.
    *   **Two-Sentence Prompts (`split_sentences`):** The outer array contains exactly two inner arrays.
*   **Mask Values:**
    *   **`1` (Character Segment):** Must contain the **Subject** + **State/Action** + **Relevant Objects**.
    *   **`0` (Context Segment):** Contains locative expressions ("On the left,"), connectors ("and"), or background descriptions. Do *not* include the safeguard text.
*   **Mask Count:** There must be exactly **two** segments marked with `1` across the entire prompt structure.
*   **Formatting:** The last segment of every inner array (sentence) must end with a period.

#### 5. Output Format
Return **only** valid JSON matching this structure exactly.

```json
{
  "safeguard_suffix": "The scene is filmed as a continuous shot with a static camera, ensuring no cuts and no new objects entering.",
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
