import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from app import preprocess_image_for_resnet

# Generate random RGB image arrays
# Heights and widths from 1 to 2000 pixels
@given(
    st.lists(
        st.integers(min_value=0, max_value=255),
        min_size=3, max_size=3
    ).flatmap(lambda rgb: st.builds(
        lambda h, w: np.full((h, w, 3), rgb, dtype=np.uint8),
        st.integers(min_value=1, max_value=2000),
        st.integers(min_value=1, max_value=2000)
    ))
)
@settings(max_examples=50, deadline=None)
def test_preprocess_image_for_resnet_property(image_array):
    """
    Test that preprocess_image_for_resnet always returns a tensor
    of shape (1, 3, 224, 224) regardless of the input image dimensions.
    """
    try:
        tensor = preprocess_image_for_resnet(image_array)
        assert tensor.shape == (1, 3, 224, 224)
    except Exception as e:
        pytest.fail(f"Preprocessing failed for image of shape {image_array.shape}: {e}")

@given(
    st.lists(
        st.integers(min_value=0, max_value=255),
        min_size=1, max_size=1
    ).flatmap(lambda gray: st.builds(
        lambda h, w: np.full((h, w), gray[0], dtype=np.uint8),
        st.integers(min_value=1, max_value=2000),
        st.integers(min_value=1, max_value=2000)
    ))
)
@settings(max_examples=20, deadline=None)
def test_preprocess_image_for_resnet_grayscale(image_array):
    """
    Test how the preprocessing handles grayscale images (2 dimensions).
    It should either handle them gracefully or throw a recognizable error,
    but since ToPILImage handles grayscale by making a 1-channel tensor,
    let's see what it outputs. Actually, ResNet requires 3 channels.
    """
    try:
        tensor = preprocess_image_for_resnet(image_array)
        # ToPILImage on (H, W) array creates mode 'L' image
        # ToTensor converts it to (1, H, W)
        assert tensor.shape == (1, 1, 224, 224)
    except Exception as e:
        pytest.fail(f"Preprocessing failed for grayscale image of shape {image_array.shape}: {e}")
