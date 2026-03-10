from .agents import TextAgent, VideoHost
from .robust_parser import RobustParser


def evaluate_video(engine, video_path, target_prompt):
    # print(f"\n=== Evaluating: {os.path.basename(video_path)} ===")

    # Initialize
    host = VideoHost(engine, video_path)
    agent1 = TextAgent("Assistant-One", "Assistant-one", engine)
    agent2 = TextAgent("Assistant-Two", "Assistant-two", engine)

    # --- Step 1: Blind Observation ---
    # print(">> Step 1: Blind Observation")
    description = host.blind_observation()
    # print(f"   [Observation]: {description[:100]}...")

    # --- Step 2: Assistants Generate Questions ---
    # print(">> Step 2: Agents Critique")
    q1 = agent1.generate_questions(target_prompt, description)
    q2 = agent2.generate_questions(target_prompt, description)
    combined_questions = f"{q1}\n{q2}"
    # print(f"   [Critique]: {combined_questions[:100]}...")

    # --- Step 3: Reflection ---
    # print(">> Step 3: Host Reflection")
    # Returns the 'Descriptions' section as requested
    reflection_info = host.reflection(target_prompt, combined_questions)

    # --- Step 4: Final Judgment ---
    # print(">> Step 4: Final Judgment")
    raw_result = host.final_score(target_prompt, description, reflection_info)

    # Extract Score
    score = RobustParser.extract_score(raw_result)
    # print(f"Final Score: {score}/3")
    # print(f"   [Reasoning]: {raw_result[:200]}...")

    # return {"video": video_path, "score": score, "full_reasoning": raw_result}
    return score
