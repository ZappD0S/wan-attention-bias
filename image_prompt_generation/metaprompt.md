**Role:** You are an expert dataset generator for AI video synthesis.
**Task:** Generate a JSON file containing 10 distinct scene datasets.
**Subject Matter:** Scenes depicting two similar or identical characters (and optional inanimate objects) against simple/neutral backgrounds.

#### 1. Scene & Character Constraints
*   **Characters:** Exactly two characters per scene. They must be similar or identical.
*   **Background:** Must be simple, neutral, or out of focus.
*   **Independence:** Characters must never interact with each other physically.
*   **Movement:** Actions must be dynamic but stationary (e.g., eating, typing, flashing a light).
*   **State Physics & Causality (CRITICAL):**
    *   **Precondition State (The "Enabling" State):** If a character or object has multiple physical states (e.g., open/closed, extended/retracted) and the action requires a specific state to function, **you must explicitly describe that state in the appearance prompt.**
        *   *Example:* Action "triggering a flash" -> Appearance "camera with the flash unit **raised/open**".
        *   *Example:* Action "typing" -> Appearance "laptop is **open**".
    *   **Anti-Result State (The "Starting" State):** If the action *changes* the state of an object, the appearance prompt must describe the state *before* the change.
        *   *Example:* Action "opening a door" -> Appearance "door is **closed**".
        *   *Example:* Action "inflating a balloon" -> Appearance "balloon is **deflated**".

#### 2. The Prompt Types
For each of the 10 scenes, generate 5 distinct prompts. **Plan Actions X and Y first**, then reverse-engineer the `appearance_prompt`.

1.  **`appearance_prompt`** (Scene Setup): Describes the visual state of characters and objects *before* movement begins.
    *   *Constraint:* **Action Readiness.** The scene is staged for the action to start immediately.
    *   *Constraint:* **Explicit Configuration.** Describe the necessary mechanical/physical state (e.g., "holding an open book," "standing next to a closed door").
    *   *Constraint:* **Orientation.** Ensure the relevant features (screens, buttons, ports) are facing the camera.
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
*   *Action X:* Taking a photo with a flash (Needs: Camera, **Flash Unit Raised/Open**).
*   *Action Y:* Reviewing a photo on the screen (Needs: Camera, **Back/Screen Visible**).
*   *Resulting Appearance:* Left char holds a camera with the **flash pop-up open**; Right char holds a camera **turned to show the screen**.

```json
[
  {
    "appearance_prompt": {
      "segments": ["On the left,", "a photographer holds a DSLR camera with the flash unit raised", "and on the right,", "a photographer holds a DSLR camera turned to show the back screen", "in a studio."],
      "mask": [0, 1, 0, 1, 0]
    },
    "action_prompts": {
      "default": {
        "segments": ["On the left,", "the photographer is triggering a bright flash from the camera", "while on the right,", "the photographer is scrolling through photos on the screen."],
        "mask": [0, 1, 0, 1]
      },
      "first_action": {
        "segments": ["On the left,", "the photographer is triggering a bright flash from the camera", "and on the right,", "the photographer is also triggering a bright flash from a camera."],
        "mask": [0, 1, 0, 1]
      },
      "second_action": {
        "segments": ["On the left,", "the photographer is scrolling through photos on the screen", "and on the right,", "the photographer is also scrolling through photos on the screen."],
        "mask": [0, 1, 0, 1]
      },
      "no_locative": {
        "segments": ["A photographer is triggering a bright flash from a camera", "and", "a photographer is scrolling through photos on a screen."],
        "mask": [1, 0, 1]
      }
    }
  }
  // ... Repeat for 10 items
]
```
