from .prompts import ACTION_PROMPTS
from .robust_parser import RobustParser


class TextAgent:
    def __init__(self, name, prompt_key, engine):
        self.name = name
        self.system_prompt = ACTION_PROMPTS[prompt_key]
        self.engine = engine

    def generate_questions(self, target_prompt, current_description):
        user_text = (
            f"Target Text Prompt: {target_prompt}\n"
            f"Current Video Description: {current_description}\n"
            "Please analyze and ask your questions."
        )
        # Call engine without video_path -> Text Only mode
        raw_response = self.engine.generate(self.system_prompt, user_text, video_path=None)

        question = RobustParser.extract_question(raw_response)
        return f"Question from {self.name}: {question}"


class VideoHost:
    def __init__(self, engine, video_path):
        self.engine = engine
        self.video_path = video_path

    def blind_observation(self):
        user_text = "Watch this video and provide the caption and description."
        raw = self.engine.generate(ACTION_PROMPTS["gpt4o-system"], user_text, self.video_path)
        return RobustParser.extract_section(raw, "Video Description")

    def reflection(self, target_prompt, assistant_questions):
        user_text = (
            f"Target Text Prompt: {target_prompt}\nQuestions to answer:\n{assistant_questions}"
        )
        raw = self.engine.generate(ACTION_PROMPTS["gpt4o-answer"], user_text, self.video_path)

        # FIX:
        # Matches original code logic: We want the 'Description' part of the answer
        # to pass into the final history, not just the answers.
        desc_part = RobustParser.extract_section(raw, "Descriptions")

        #
        # # Fallback: If 'Descriptions' is missing, try 'Answers'
        # if not desc_part:
        #     desc_part = RobustParser.extract_section(raw, "Answers")

        # Final Fallback: Return raw text
        if not desc_part:
            print("[Parsing Warning] Failed to extract 'Descriptions'. Returning full text.")
            desc_part = raw

        return desc_part

    def final_score(self, target_prompt, description, reflection):
        user_text = (
            f"Target Text Prompt: {target_prompt}\n"
            f"Info 1 (Objective): {description}\n"
            f"Info 2 (Reflection): {reflection}\n"
            "Provide the final Updated Video Description and Evaluation Result."
        )
        return self.engine.generate(ACTION_PROMPTS["summer-system"], user_text, self.video_path)
