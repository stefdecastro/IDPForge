"""
utils/smart_scoring.py
"""
def get_smart_threshold(attempt, current_count, base=10.0, inc=5.0):
    multiplier = attempt // 100 if current_count == 0 else attempt // 500
    return base + (multiplier * inc)