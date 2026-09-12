I have some scenes that are used for video generation stored as a json file.
These scenes were generated using the prompt in the file metaprompt.md.

I need to generate 150-200 of them in total. If I ask an LLM to generate all at once maybe it's too much. Maybe we should generate them in different batches. But if we do so, we risk having some redundancies or scenes that are excessively similar between each other. So in a second step I want the LLM to go through the generated scenes and fix the redundancies by replacing one of the scenes with a new one. This will be repeated until no redundancies are detected. How do you recommend describing this process?
