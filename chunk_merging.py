def find_overlap(chunk1, chunk2):
    """
    Returns chunk2 with any leading overlapping text removed based on chunk1.
    """
    # Strip whitespace for clean comparison
    chunk1 = chunk1.strip()
    chunk2 = chunk2.strip()

    # We're going to look for longest tail of chunk1 that appears at start of chunk2
    max_overlap = ""
    for i in range(len(chunk1)):
        # Try slicing the chunk1 from position i to the end
        tail = chunk1[i:]

        # If chunk2 starts with this tail, we found an overlap!
        if chunk2.startswith(tail):
            max_overlap = tail
            break  # Longest match is found; no need to keep going

    # Remove the overlapping prefix from chunk2
    if max_overlap:
        return chunk2[len(max_overlap):]
    # If no overlap, return chunk2 unchanged
    return chunk2
