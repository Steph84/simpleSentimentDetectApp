import pytest

from app import process_tweet

def test_positive():
	phrase_pos = '@ashlili LIKE MEEEEEEEEE '
	res = process_tweet(phrase_pos)
	assert res == "positif"


def test_negative():
	phrase_neg = '@ home studying for maths wooot ! im so going to fail this shit '
	res = process_tweet(phrase_neg)
	assert res == "négatif"