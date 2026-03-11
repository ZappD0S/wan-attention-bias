import re


class RobustParser:
    @staticmethod
    def extract_section(text: str, tag_name: str) -> str:
        """
        Robustly extracts content following a tag like [Video Description]:
        """
        if not text:
            raise ValueError(f"Input text for extracting '{tag_name}' is empty.")

        # Pattern 1: Strict w/ optional Markdown (e.g., **[Video Description]:**)
        # grab everything until you see a new line that starts with [ (like [Answers]), or until the string ends
        pattern_strict = rf"(?:\*\*|)?\[{tag_name}\](?:\*\*|)?:\s*(.*?)(?=\n(?:\*\*|)?\[|$)"
        match = re.search(pattern_strict, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 2: Loose (No brackets) (e.g., Video Description:)
        pattern_loose = rf"{tag_name}:\s*(.*)"
        match = re.search(pattern_loose, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Pattern 3: Fallback - Look for the tag appearing anywhere and take the rest
        if tag_name.lower() in text.lower():
            start_index = text.lower().find(tag_name.lower())
            colon_index = text.find(":", start_index)
            if colon_index != -1:
                return text[colon_index + 1 :].strip()

        # Final Fallback: Return full text but warn the user
        print(f"[Parsing Warning] Failed to extract tag '{tag_name}'. Returning full text.")
        return text

    @staticmethod
    def extract_question(text: str) -> str:
        """Extracts content inside <question> tags."""
        if not text:
            return ""  # Questions are optional, so returning empty string here is usually safer

        match = re.search(r"<question>\s*(.*?)\s*</question>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def extract_score(text: str) -> int:
        """
        Robustly looks for the score.
        - Handles words ("three" -> 3)
        - Handles floats (2.9 -> 3)
        - Clamps values to [1, 3] range
        - Raises ValueError on failure or empty input
        """
        if not text:
            raise ValueError("Input text for score extraction is empty.")

        text_lower = text.lower()

        # 1. Pre-processing: Convert common words to digits
        word_map = {"one": "1", "two": "2", "three": "3", "poor": "1", "moderate": "2", "good": "3"}
        for word, digit in word_map.items():
            text_lower = re.sub(rf"\b{word}\b", digit, text_lower)

        # 2. Define Regex Strategies (Priority Ordered)
        patterns = [
            # Strategy A: Explicit "Evaluation Result: X"
            r"Evaluation Result.*?:.*?(\d+(?:\.\d+)?)",
            # Strategy B: "X because" (Standard prompt format)
            r"(?<!\d|\.)(\d+(?:\.\d+)?)\s*(?:,|\.)?\s*because",
            # Strategy C: "Score: X", "Rating: X"
            r"(?:score|rating|consistency)\s*[:=-]\s*(\d+(?:\.\d+)?)",
            # Strategy D: "X/3" or "X out of 3"
            r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*3",
        ]

        detected_val = None

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    detected_val = float(match.group(1))
                    break
                except ValueError:
                    continue

        # 3. Final Fallback: Look for any isolated digit 1-3 in the last line
        if detected_val is None:
            lines = text_lower.strip().split("\n")
            if lines:
                last_line = lines[-1]
                match = re.search(r"\b(\d+(?:\.\d+)?)\b", last_line)
                if match:
                    detected_val = float(match.group(1))

        # 4. Process Score
        if detected_val is not None:
            # Round (2.6 -> 3) and Clamp (0 -> 1, 5 -> 3)
            return max(1, min(3, round(detected_val)))

        # 5. Failure
        raise ValueError(f"Could not extract a valid score. Text: '{text}...'")
