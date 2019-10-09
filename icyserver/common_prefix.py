
from collections import defaultdict


def find_common_prefix(candidates: [str], min_width: int, depth: int, nth=0) -> str:
    """This function finds the common prefix for all candidates.
    Args:
        candidates: List of candidate strings.
        min_width: Minimum count of common prefixed candidates.
        depth: Maximum count of chars for common prefix.
    Returns:
        The common prefix.
    Examples:

    >>> find_common_prefix(['123456', '12456', '4321'], min_width=2, depth=1)
    '1'
    >>> find_common_prefix(['123456', '12456', '4321', '12356'], min_width=2, depth=2)
    '12'
    >>> find_common_prefix(['123456', '12456', '4321', '12356'], min_width=2, depth=3)
    '123'
    >>> find_common_prefix(['123456', '12456', '4321', '12356'], min_width=3, depth=3)
    '12'
    >>> find_common_prefix(['123456', '12456', '4321', '12356'], min_width=4, depth=1)
    ''
    """

    if len(candidates) < min_width or nth >= depth:
        return None

    reduced = defaultdict(list)
    for i, candidate in enumerate(candidates):
        if len(candidate) == 0:
            continue
        reduced[candidate[0]].append(candidate[1:])

    wid, k, sub_candidates = max([(len(v), k, v) for k, v in reduced.items()])
    if wid < min_width:
        return None
    else:
        n = find_common_prefix(sub_candidates, min_width, depth, nth+1)
        return [k] + (n or [])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
