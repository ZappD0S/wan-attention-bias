import logging
import re
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path

from .engine import QwenEngine

logger = logging.getLogger(__name__)


class VideoAsset:
    """Represents a video segment. Lazily generates a description only if needed."""

    def __init__(self, path: Path, engine: QwenEngine):
        self.path = path
        self.engine = engine
        self._description: str | None = None

    @property
    def description(self) -> str:
        if self._description is None:
            prompt = textwrap.dedent("""\
                You are a forensic video analyst.
                Provide a detailed, objective, chronological log of the video.

                Guidelines:
                1. Break the video down by visual changes and movements.
                2. Describe specific body parts, objects, and interactions.
                3. Do NOT interpret intent or purpose.
                4. Focus purely on visual observables.

                Output the log now.""")
            self._description = self.engine.generate(prompt, video_path=self.path)
        return self._description


class ActionAuditor(ABC):
    def __init__(self, engine: QwenEngine):
        self.engine = engine

    @abstractmethod
    def _score(self, video: VideoAsset, action: str) -> float:
        pass

    def score_pair(self, video: VideoAsset, action_a: str, action_b: str) -> float:
        score_a = self._score(video, action_a)
        score_b = self._score(video, action_b)
        return score_a - score_b

    @staticmethod
    def _extract_score(model_output: str) -> int:
        if not model_output:
            raise ValueError("Model output is empty.")

        # the * are just in case the the model uses bold formatting for Score
        matches = re.findall(r"Score\**\s*:\s*(\d)", model_output, re.IGNORECASE)

        if not matches:
            raise ValueError(
                f"Strict parsing failed. Could not find 'Score: X' in model output:\n{model_output}"
            )

        # take the last match to ensure we get the final conclusion, and avoid any scores in the reasoning
        final_score = int(matches[-1])

        # clamp the value between 1 and 5 in case the model hallucinates a weird number
        return max(1, min(5, final_score))


class SoftDirectAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt1 = textwrap.dedent(f"""\
            You are a strict video auditor.
            Analyze the video content and determine if the following action occurs.

            Target Action: {action}

            Provide a step-by-step reasoning based on the visual evidence.
            Conclude by evaluating how well the video matches the action.""")

        reasoning = self.engine.generate(prompt1, video_path=video.path)
        logger.debug("[SoftDirectAuditor] Reasoning text generated:\n---\n%s\n---", reasoning)

        prompt2 = textwrap.dedent("""\
            Based on your reasoning, assign a match score on a scale of 1 to 5.

            Criteria:
            5: Perfect Match (Action is clearly the main focus).
            4: Strong Match (Action occurs, minor noise).
            3: Partial Match (Action is part of a larger sequence).
            2: Weak Match (Ambiguous or hard to see).
            1: No Match (Action does not happen).

            CRITICAL INSTRUCTION: Output ONLY a single integer (1, 2, 3, 4, or 5). Do not output any words, punctuation, or spaces.""")

        logits = self.engine.get_scoring_logits(prompt1, reasoning, prompt2, video.path)
        return self.engine.calculate_soft_score(
            logits,
            target_tokens=["1", "2", "3", "4", "5"],
            target_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
        )


class SoftTwoAFCAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        raise NotImplementedError(
            "SoftTwoAFCAuditor is a pairwise evaluator and cannot score a single action. "
            "Please call `score_pair(video, action_a, action_b)` instead."
        )

    def score_pair(self, video: VideoAsset, action_a: str, action_b: str) -> float:
        prompt1 = textwrap.dedent(f"""\
            You are an expert video evaluator.
            I will provide you with a short video clip and two possible descriptions of the action occurring in the video.

            Option A: {action_a}
            Option B: {action_b}

            Your task is to determine which of the two options best describes the action and movement being performed by the character in the video.

            CRITICAL INSTRUCTIONS:
            1. Focus strictly on the action, movement, and physical interactions.
            2. Do NOT base your decision solely on the character's clothing, background, or static objects. Focus on what the character is doing.
            3. You MUST choose either Option A or Option B.

            Provide a step-by-step reasoning based on the visual evidence comparing the two options.""")

        reasoning = self.engine.generate(prompt1, video_path=video.path)

        prompt2 = textwrap.dedent("""\
            Based on your reasoning, choose the option that best matches the video.

            CRITICAL INSTRUCTION: Output ONLY a single uppercase letter ("A" or "B"). Do not output any words, punctuation, or spaces.""")

        logits = self.engine.get_scoring_logits(prompt1, reasoning, prompt2, video.path)

        prob_a = self.engine.calculate_soft_score(
            logits, target_tokens=["A", "B"], target_weights=[1.0, 0.0]
        )
        prob_b = 1.0 - prob_a

        # Return the margin.
        # If prob_a is 0.9 and prob_b is 0.1, it returns 0.8.
        return prob_a - prob_b


class SoftBlindAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt1 = textwrap.dedent(f"""\
            You are a strict action auditor.
            Your task is to rate how well the 'Target Action' matches the 'Video Description'.

            Target Action: {action}
            Video Description: {video.description}

            Reasoning Rules:
            - Rely ONLY on the Video Description provided above. 
            - Be skeptical: if a specific detail is missing from the description, assume it did not happen.
            - Check for chronological consistency.

            Output a brief step-by-step reasoning.""")

        reasoning = self.engine.generate(prompt1, None)

        prompt2 = textwrap.dedent("""\
            Based on the reasoning above, assign a match score on a scale of 1 to 5.

            Criteria:
            5: Perfect Match (Unambiguous, clearly main focus).
            4: Strong Match (Main event, minor noise).
            3: Partial Match (Action occurred but mixed with others).
            2: Weak Match (Ambiguous or minor detail).
            1: No Match (Action not found or different action).

            CRITICAL INSTRUCTION: Output ONLY a single integer (1, 2, 3, 4, or 5). Do not output any words, punctuation, or spaces.""")

        logits = self.engine.get_scoring_logits(prompt1, reasoning, prompt2, None)
        return self.engine.calculate_soft_score(
            logits,
            target_tokens=["1", "2", "3", "4", "5"],
            target_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
        )


class DiscreteDirectAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt = textwrap.dedent(f"""\
            You are a strict action auditor and forensic video analyst.
            Your task is to determine if the 'Target Action' occurs in the video based on visual evidence.

            Target Action: {action}

            Reasoning Rules:
            - Focus purely on visual observables (movements, contacts, states).
            - Do NOT interpret intent or purpose.
            - Be skeptical: if the specific visual details are missing, assume it did not happen.

            Scoring Criteria:
            5: Perfect Match (Unambiguous, clearly main focus).
            4: Strong Match (Main event, minor noise).
            3: Partial Match (Action occurred but mixed with others).
            2: Weak Match (Ambiguous or minor detail).
            1: No Match (Action not found or different action).

            Instructions:
            1. Output a brief step-by-step reasoning based on the visual evidence.
            2. End your response strictly with: "Score: X" (where X is 1-5).""")

        return self._extract_score(self.engine.generate(prompt, video_path=video.path))


class DiscreteBlindAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt = textwrap.dedent(f"""\
            You are a strict action auditor.
            Your task is to rate how well the 'Target Action' matches the 'Video Description'.

            Target Action: {action}
            Video Description: {video.description}

            Reasoning Rules:
            - Rely ONLY on the Video Description provided above. 
            - Be skeptical: if a specific detail is missing from the description, assume it did not happen.
            - Check for chronological consistency.

            Scoring Criteria:
            5: Perfect Match (Unambiguous, clearly main focus).
            4: Strong Match (Main event, minor noise).
            3: Partial Match (Action occurred but mixed with others).
            2: Weak Match (Ambiguous or minor detail).
            1: No Match (Action not found or different action).

            Instructions:
            1. Output a brief step-by-step reasoning.
            2. End your response strictly with: "Score: X" (where X is 1-5).""")

        return self._extract_score(self.engine.generate(prompt, None))
