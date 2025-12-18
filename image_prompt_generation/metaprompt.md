**Role:** You are an expert dataset generator for AI video synthesis.
**Task:** Generate a JSON file containing 10 distinct scene datasets.
**Subject Matter:** Scenes depicting two similar or identical characters (and optional inanimate objects) against simple/neutral backgrounds.

#### 1. Scene & Character Constraints
*   **Characters:** Exactly two characters per scene. They must be similar or identical (e.g., two identical robots, two similar cats).
*   **Background:** Must be simple, neutral, or out of focus to ensure characters are the primary subject.
*   **Independence:** Characters must never interact with each other physically. They act independently.
*   **Movement:** Actions must be dynamic but stationary (e.g., eating, waving, typing) rather than locomotive (e.g., running away). The characters must stay largely in place.
*   **Realism:** Actions must be physically possible for the specific character type.

#### 2. The Prompt Types
For each of the 10 scenes, you must generate 5 distinct prompts based on the logic below:

1.  **`appearance_prompt`**: Describes **only** the visual appearance of the characters and the scene. No actions allowed.
    *   *Note:* If an action requires a specific body part (e.g., kicking requires visible legs), ensure that body part is described here.
2.  **`default`** (Action Prompt 1): Character A performs **Action X**. Character B performs **Action Y**.
    *   *Constraint:* Actions X and Y must be distinct.
    *   *Constraint:* Must use locative expressions (e.g., "on the left", "in the background") to position characters.
3.  **`first_action`** (Action Prompt 2): Character A performs **Action X**. Character B performs **Action X**.
    *   *Constraint:* Both perform the first action defined in the `default` prompt.
    *   *Constraint:* Must use locative expressions.
4.  **`second_action`** (Action Prompt 3): Character A performs **Action Y**. Character B performs **Action Y**.
    *   *Constraint:* Both perform the second action defined in the `default` prompt.
    *   *Constraint:* Must use locative expressions.
5.  **`no_locative`** (Action Prompt 4): Character A performs **Action X**. Character B performs **Action Y**.
    *   *Constraint:* Same distinct actions as `default`.
    *   *Constraint:* **STRICT FORBIDDEN:** Do not use any locative/spatial words (e.g., "left", "right", "next to").

#### 3. Segmentation & Masking Rules
Every prompt must be split into a `segments` array and a `mask` array.
*   **Reconstruction:** `" ".join(segments)` must create a grammatically correct, natural sentence.
*   **Character Segments (Mask = 1):**
    *   Must contain the **Subject** + **Description/Action**.
    *   Must make sense on its own (No ambiguous pronouns like "the other one").
    *   *Example:* "a robot is waving" (Valid). "is waving" (Invalid).
*   **Context Segments (Mask = 0):**
    *   Contains locative expressions ("On the left,"), connectors ("and", "while"), or background descriptions.
    *   *Rule:* Locative expressions must **never** be inside a character segment (Mask 1). They must be their own segment (Mask 0).
*   **Mask Count:** There must be exactly two `1`s in the mask array (one for each character).

#### 4. Output Format
Return **only** valid JSON matching this structure exactly:

```json
[
  {
    "appearance_prompt": {
      "segments": ["On the left,", "a fluffy white cat sits", "and on the right,", "another fluffy white cat sits", "against a grey wall."],
      "mask": [0, 1, 0, 1, 0]
    },
    "action_prompts": {
      "default": {
        "segments": ["On the left,", "the fluffy white cat is grooming its paw", "while on the right,", "the fluffy white cat is yawning widely."],
        "mask": [0, 1, 0, 1]
      },
      "first_action": {
        "segments": ["On the left,", "the fluffy white cat is grooming its paw", "and on the right,", "the fluffy white cat is also grooming its paw."],
        "mask": [0, 1, 0, 1]
      },
      "second_action": {
        "segments": ["On the left,", "the fluffy white cat is yawning widely", "and on the right,", "the fluffy white cat is also yawning widely."],
        "mask": [0, 1, 0, 1]
      },
      "no_locative": {
        "segments": ["A fluffy white cat is grooming its paw", "and", "a fluffy white cat is yawning widely."],
        "mask": [1, 0, 1]
      }
    }
  }
  // ... Repeat for 10 items
]
```

