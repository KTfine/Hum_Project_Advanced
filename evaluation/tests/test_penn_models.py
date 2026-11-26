import numpy as np
import pytest
from types import SimpleNamespace

from evaluation.data.penn_action import PENN_ACTIONS
from evaluation.models.penn_deepseek import validate_penn_prediction
from evaluation.models.pose import coco17_to_penn13
from evaluation.models.xclip import XCLIP_VARIANTS, XClipPennClassifier, penn_text_prompts


def _valid_deepseek_value():
    return {
        "activity_probabilities": {
            label: (1.0 if label == "bowl" else 0.0)
            for label in PENN_ACTIONS
        },
        "frames": [
            {"joints": [[0.5, 0.5] for _ in range(13)]}
            for _ in range(8)
        ],
    }


def test_xclip_ladder_and_prompts_are_frozen_to_penn_labels():
    assert XCLIP_VARIANTS == (
        ("large-patch14", "microsoft/xclip-large-patch14"),
        ("base-patch32", "microsoft/xclip-base-patch32"),
    )
    prompts = penn_text_prompts()
    assert len(prompts) == 15
    assert prompts[0] == "a video of a person performing a baseball pitch"
    assert prompts[8] == "a video of a person performing pull ups"


def test_deepseek_penn_parser_normalizes_probabilities_and_pose_shape():
    parsed = validate_penn_prediction(_valid_deepseek_value())
    assert parsed["activity_probabilities"][PENN_ACTIONS.index("bowl")] == 1.0
    assert len(parsed["frames"]) == 8
    assert len(parsed["frames"][0]["joints"]) == 13


def test_deepseek_penn_parser_rejects_missing_label_and_bad_pose():
    value = _valid_deepseek_value()
    del value["activity_probabilities"][PENN_ACTIONS[0]]
    with pytest.raises(ValueError, match="exactly the 15"):
        validate_penn_prediction(value)
    value = _valid_deepseek_value()
    value["frames"][0]["joints"] = value["frames"][0]["joints"][:-1]
    with pytest.raises(ValueError, match="13 joints"):
        validate_penn_prediction(value)


def test_coco17_to_penn13_preserves_order_and_marks_missing_joints():
    xy = np.asarray([[index * 10.0, index * 5.0] for index in range(17)])
    confidence = np.ones(17)
    confidence[7] = 0.01
    joints, scores = coco17_to_penn13(
        xy, confidence, width=160, height=80, confidence_threshold=0.05,
    )
    assert joints[0] == [0.0, 0.0]
    assert joints[1] == [0.3125, 0.3125]
    assert joints[3] is None
    assert scores[3] == 0.01
    assert len(joints) == len(scores) == 13


def test_xclip_processor_receives_frames_as_images(tmp_path):
    from PIL import Image

    paths = []
    for index in range(8):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (16, 16), "white").save(path)
        paths.append(path)

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        @property
        def shape(self):
            return self.value.shape

        def to(self, _device):
            return self

        def __getitem__(self, item):
            return Tensor(self.value[item])

        def float(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return InferenceMode()

        @staticmethod
        def softmax(_value, dim):
            assert dim == -1
            return Tensor(np.full(15, 1 / 15))

    class Processor:
        def __call__(self, **kwargs):
            assert "images" in kwargs
            assert "videos" not in kwargs
            assert len(kwargs["images"]) == 8
            return {"pixel_values": Tensor(np.zeros((1, 8, 3, 2, 2)))}

    class Model:
        def __call__(self, **kwargs):
            assert kwargs["pixel_values"].shape == (1, 8, 3, 2, 2)
            return SimpleNamespace(logits_per_video=Tensor(np.zeros((1, 15))))

    classifier = XClipPennClassifier.__new__(XClipPennClassifier)
    classifier.torch = FakeTorch()
    classifier.device = "cpu"
    classifier.processor = Processor()
    classifier.model = Model()
    probabilities = classifier.predict(paths)
    assert len(probabilities) == 15
    assert sum(probabilities) == pytest.approx(1.0)
