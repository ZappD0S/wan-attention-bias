from evaluation import ActionAuditor

from .agents import TextAgent, VideoHost
from .robust_parser import RobustParser


class VideoBenchAuditor(ActionAuditor):
    def _score(self, video, action: str) -> float:
        host = VideoHost(self.engine, video.path)
        agent1 = TextAgent("Assistant-One", "Assistant-one", self.engine)
        agent2 = TextAgent("Assistant-Two", "Assistant-two", self.engine)

        description = host.blind_observation()

        q1 = agent1.generate_questions(action, description)
        q2 = agent2.generate_questions(action, description, previous_question=q1)

        reflection_info = host.reflection(action, q1, q2)

        raw_result = host.final_score(action, description, reflection_info)

        score = RobustParser.extract_score(raw_result)

        return float(score)
