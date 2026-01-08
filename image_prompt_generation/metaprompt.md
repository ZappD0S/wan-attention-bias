**Role:** You are an expert dataset generator for AI video synthesis.
**Task:** Generate a JSON file containing 10 distinct scene datasets.
**Subject Matter:** Scenes depicting two similar or identical characters (and optional inanimate objects) against simple/neutral backgrounds.

#### 1. Scene & Character Constraints
*   **Characters:** Exactly two characters per scene. They must be similar or identical.
*   **Background:** Must be simple, neutral, or out of focus.
*   **Independence:** Characters must never interact with each other physically.
*   **Movement:** Actions must be dynamic but stationary (e.g., eating, typing, flashing a light).
*   **State Physics & Causality (CRITICAL UPDATES):**
    *   **Universal Compatibility:** Since the dataset includes prompts where *both* characters perform Action X and *both* perform Action Y, **Action X and Action Y must share a compatible physical starting state.**
        *   *Invalid:* Action X = "Opening a closed box" / Action Y = "Closing an open box" (Contradictory starting states).
        *   *Valid:* Action X = "Typing on laptop" / Action Y = "Closing laptop" (Both start with laptop open).
    *   **Identical Initial State:** The `appearance_prompt` must describe **both characters exactly the same way**. They must both hold the same objects in the same configuration, ready for *either* action to occur.
    *   **Precondition State:** You must explicitly describe the state required to enable the actions.
        *   *Example:* Actions "Taking a selfie" & "Texting" -> Appearance "Both holding a smartphone with the **screen facing them**."

#### 2. The Prompt Types
For each of the 10 scenes, generate 5 distinct prompts. **Plan Actions X and Y first**, then reverse-engineer the `appearance_prompt`.

1.  **`appearance_prompt`** (Scene Setup): Describes the visual state of characters and objects *before* movement begins.
    *   *Constraint:* **Symmetry.** Character A and Character B must be described with identical states/objects.
    *   *Constraint:* **Action Readiness.** The setup must allow *either* Action X or Action Y to start immediately without a cut.
    *   *Constraint:* **Passive Verbs.** Use state verbs (holding, facing, wearing, resting).
2.  **`default`** (Action Prompt 1): Character A performs **Action X**. Character B performs **Action Y**.
    *   *Constraint:* Distinct actions. Must use locative expressions.
3.  **`first_action`** (Action Prompt 2): Character A performs **Action X**. Character B performs **Action X**.
    *   *Constraint:* Both perform Action X. Must use locative expressions.
4.  **`second_action`** (Action Prompt 3): Character A performs **Action Y**. Character B performs **Action Y**.
    *   *Constraint:* Both perform Action Y. Must use locative expressions.
5.  **`no_locative`** (Action Prompt 4): Character A performs **Action X**. Character B performs **Action Y**.
    *   *Constraint:* Same actions as `default`, but **NO** spatial words allowed.

#### 3. Segmentation & Masking Rules
Every prompt must be split into a `segments` array and a `mask` array.
*   **Reconstruction:** `" ".join(segments)` must create a grammatically correct sentence.
*   **Character Segments (Mask = 1):**
    *   Must contain the **Subject** + **State/Action** + **Relevant Objects/Features**.
    *   *Rule:* Include the object being interacted with in this segment.
*   **Context Segments (Mask = 0):**
    *   Contains locative expressions ("On the left,"), connectors ("and"), or background descriptions.
*   **Mask Count:** Exactly two `1`s in the mask array.

#### 4. Output Format
Return **only** valid JSON matching this structure exactly.

**Example Logic:**
*   *Action X:* Eating a burger (Needs: Holding burger near face).
*   *Action Y:* Wiping mouth with a napkin (Needs: Holding burger, holding napkin).
*   *Required Start State:* Both characters holding a burger in one hand and a napkin in the other. (Compatible with both X and Y).

```json
[
  {
    "appearance_prompt": {
      "segments": ["On the left,", "a diner holds a large burger in one hand and a napkin in the other", "and on the right,", "a diner holds a large burger in one hand and a napkin in the other", "seated at a table."],
      "mask": [0, 1, 0, 1, 0]
    },
    "action_prompts": {
      "default": {
        "segments": ["On the left,", "the diner is taking a large bite out of the burger", "while on the right,", "the diner is wiping their mouth with the white napkin."],
        "mask": [0, 1, 0, 1]
      },
      "first_action": {
        "segments": ["On the left,", "the diner is taking a large bite out of the burger", "and on the right,", "the diner is also taking a large bite out of the burger."],
        "mask": [0, 1, 0, 1]
      },
      "second_action": {
        "segments": ["On the left,", "the diner is wiping their mouth with the white napkin", "and on the right,", "the diner is also wiping their mouth with the white napkin."],
        "mask": [0, 1, 0, 1]
      },
      "no_locative": {
        "segments": ["A diner is taking a large bite out of a burger", "and", "a diner is wiping their mouth with a white napkin."],
        "mask": [1, 0, 1]
      }
    }
  }
  // ... Repeat for 10 items
]
```
