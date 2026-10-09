import unittest

import torch
from torch import nn

from evaluate import metrics
from verifai.agents import (
    DecisionVerificationAgent,
    InputQualityAgent,
    ModelSelectionAgent,
    QualityReport,
)
from verifai.attacks import fgsm, pgd
from verifai.models import DeepfakeClassifier


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        self.images = torch.rand(2, 1, 2, 2)
        self.labels = torch.tensor([0, 1])

    def test_fgsm_respects_l_infinity_budget_and_pixel_range(self):
        epsilon = 0.05
        attacked = fgsm(self.model, self.images, self.labels, epsilon)
        self.assertTrue(torch.all((attacked - self.images).abs() <= epsilon + 1e-6))
        self.assertTrue(torch.all((attacked >= 0) & (attacked <= 1)))

    def test_pgd_respects_l_infinity_budget_and_pixel_range(self):
        epsilon = 0.08
        attacked = pgd(self.model, self.images, self.labels, epsilon, 0.02, 4)
        self.assertTrue(torch.all((attacked - self.images).abs() <= epsilon + 1e-6))
        self.assertTrue(torch.all((attacked >= 0) & (attacked <= 1)))

    def test_binary_metrics_and_auc(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["false_positive_rate"], 0.0)

    def test_decision_is_inconclusive_for_detector_disagreement(self):
        decision = DecisionVerificationAgent().verify(
            {"model_a": 0.9, "model_b": 0.1}, instability=0
        )
        self.assertEqual(decision.verdict, "Inconclusive")

    def test_decision_is_inconclusive_for_sensitive_detector(self):
        decision = DecisionVerificationAgent().verify({"model_a": 0.95}, instability=0.2)
        self.assertEqual(decision.verdict, "Inconclusive")

    def test_uniform_image_is_flagged_as_low_quality(self):
        from PIL import Image

        report = InputQualityAgent().analyze(Image.new("RGB", (64, 64), color="white"))
        self.assertEqual(report.label, "low")

    def test_classifier_forward_uses_the_extracted_feature_vector(self):
        model = DeepfakeClassifier().eval()
        images = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            features = model.extract_features(images)
            logits = model(images)
            expected_logits = model.backbone.fc(features)
            self.assertEqual(features.shape, (1, 512))
            self.assertTrue(torch.equal(logits, expected_logits))

    def test_selector_uses_low_quality_specialist_when_available(self):
        class StubDetector:
            def __init__(self, profile):
                self.profile = profile

        class StubRegistry:
            names = ["balanced", "low"]

            def __getitem__(self, name):
                return StubDetector("balanced" if name == "balanced" else "low_quality")

        quality = QualityReport(128, 128, 20.0, "low")
        selection = ModelSelectionAgent().select(StubRegistry(), quality)
        self.assertEqual(selection.name, "low")


if __name__ == "__main__":
    unittest.main()
