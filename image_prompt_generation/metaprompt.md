**Role:** You are an expert dataset generator for AI video synthesis.
**Task:** Generate a JSON file containing a `safeguard_suffix` and a `dataset` of 10 distinct scenes.
**Subject Matter:** Scenes depicting two similar or identical characters (or objects) against simple/neutral backgrounds.

#### 1. Universal Safeguards (Global Constant)
For the `safeguard_suffix` key, output **exactly** this string:
> **"The scene is filmed as a continuous shot with a static camera, ensuring no cuts and no new objects entering."**

#### 2. Scene & Character Constraints
*   **Scene Setup (The First Sentence):** The `appearance_prompt` must always start with a standalone sentence that establishes the **Count** of the subjects and the **Setting/Background**. 
    *   *Goal:* Set the stage before describing individual details. 
    *   *Example:* "Two golden retrievers sit against a seamless white backdrop."
*   **Enumeration:** In the detailed segments that follow, refer to the first subject as **"one [subject]"** and the second as **"another [subject]"** or **"the second [subject]"**.
*   **Orientation:** To prevent subjects from looking at each other, describe animate characters in their specific segments as **facing the camera**.
*   **Independence:** Characters must not interact with each other physically.
*   **Movement:** Actions must be dynamic but stationary (e.g., barking, typing, signaling).

#### 3. Logical Consistency & Causality
Reverse-engineer the appearance based on the intended actions. **Do not rely on implied magic.**
*   **Preconditions:** If an action requires a specific starting state, the appearance prompt must explicitly describe it.
    *   *Example:* If a light turns on, the specific description must say "an **unlit** light bulb."
    *   *Example:* If a flash closes, the description must say "with its built-in pop-up flash **raised**."
*   **Mechanical Grounding:** Actions should be physically plausible for the specific object (e.g., rotating a head, clicking a shutter).

#### 4. The Prompt Types
For each scene, generate the `appearance_prompt` and 3 variations of action prompts.

1.  **`appearance_prompt`** (Scene Setup):
    *   **Segment 0:** Intro Sentence establishing count and background + Locative (e.g., "Two robots stand in a lab. On the left,").
    *   **Segment 1:** Specific Subject A + Orientation + State Details.
    *   **Segment 2:** Connector + Locative (e.g., "and on the right,").
    *   **Segment 3:** Specific Subject B + Orientation + State Details.
    *   **Segment 4:** Ending punctuation (".").
2.  **`default`** (Action Prompt 1): Subject A performs **Action X**. Subject B performs **Action Y** (Standard description with location).
3.  **`no_locative`** (Action Prompt 2):
    *   **Structure:** Single sentence connecting Action X and Action Y with "and".
    *   **Constraint:** Do not use spatial words (left, right, center, side, etc.).
4.  **`split_sentences`** (Action Prompt 3):
    *   **Field: `general_prompt`**: A single sentence describing the subjects and setting *without* actions.
    *   **Field: `segments`**: Two grammatically complete, independent sentences (one for Action X, one for Action Y).
    *   **Constraint:** No spatial words. Do not distinguish the characters (e.g., use "The [subject]..." for both).

#### 5. Segmentation & Masking Rules
*   **Structure:** Arrays of arrays.
*   **Mask Values:**
    *   **`1` (Subject/Action Segment):** Must contain the **Subject** + **State/Action** + **Relevant Objects**.
    *   **`0` (Context Segment):** Contains the introductory sentence, locative expressions, and connectors.
*   **Mask Count:** Exactly **two** segments marked with `1` across the entire prompt structure.

#### 6. Output Format
Return **only** valid JSON.

```json
{
  "safeguard_suffix": "The scene is filmed as a continuous shot with a static camera, ensuring no cuts and no new objects entering.",
  "dataset": [
    {
      "appearance_prompt": {
        "segments": [
          ["Two students sit at a wooden desk. On the left,", "one student sits facing the camera with an open laptop", "and on the right,", "another student sits facing the camera with an open laptop", "."]
        ],
        "mask": [
          [0, 1, 0, 1, 0]
        ]
      },
      "action_prompts": {
        "default": {
          "segments": [
            ["On the left,", "the student begins typing on the keyboard", "while on the right,", "the student closes the laptop lid slowly."]
          ],
          "mask": [
            [0, 1, 0, 1]
          ]
        },
        "no_locative": {
          "segments": [
            ["A student begins typing on a keyboard", "and", "a student closes a laptop lid slowly."]
          ],
          "mask": [
            [1, 0, 1]
          ]
        },
        "split_sentences": {
          "general_prompt": "Two students sitting facing the camera at a wooden desk with open laptops.",
          "segments": [
            ["The student begins typing on the keyboard."],
            ["The student closes the laptop lid slowly."]
          ],
          "mask": [
            [1],
            [1]
          ]
        }
      }
    }
  ]
}
```
