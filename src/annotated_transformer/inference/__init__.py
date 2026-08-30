"""Greedy decoding and qualitative evaluation of a trained model."""

from annotated_transformer.inference.decode import greedy_decode
from annotated_transformer.inference.evaluate import check_outputs

__all__ = ["greedy_decode", "check_outputs"]
