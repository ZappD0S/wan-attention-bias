from pathlib import Path

from . import QwenVLEngine
from .agents import TextAgent, VideoHost
from .robust_parser import RobustParser


def evaluate_video(engine: QwenVLEngine, video_path: str | Path, target_prompt: str):

    # Initialize
    host = VideoHost(engine, video_path)
    agent1 = TextAgent("Assistant-One", "Assistant-one", engine)
    agent2 = TextAgent("Assistant-Two", "Assistant-two", engine)

    # --- Step 1: Blind Observation ---
    description = host.blind_observation()

    # --- Step 2: Assistants Generate Questions ---
    q1 = agent1.generate_questions(target_prompt, description)
    q2 = agent2.generate_questions(target_prompt, description, previous_question=q1)

    # --- Step 3: Reflection ---
    # Returns the 'Descriptions' section as requested
    reflection_info = host.reflection(target_prompt, q1, q2)

    # --- Step 4: Final Judgment ---
    raw_result = host.final_score(target_prompt, description, reflection_info)

    # Extract Score
    score = RobustParser.extract_score(raw_result)
    return score
