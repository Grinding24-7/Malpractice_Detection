"""Definition of the custom desk-object classes.

COCO-80 has no label for paper, pen, pencil, notebook, exam sheet or desk,
which are precisely the objects that decide whether a student is still doing
their own work.  Training a small model on this list is what makes those
classes detectable; until one is trained, a clear "no object" result from the
stock model means "outside COCO's vocabulary", not "desk was clear".

Class ids are assigned in list order, so the order is part of the model's
contract -- appending is safe, reordering invalidates existing labels.
"""

from __future__ import annotations

# Index in this list == the model's class id.
DESK_CLASSES: list[str] = [
    "exam_paper",   # a sheet being written on, or blank paper within reach
    "answer_sheet", # a submitted/answer booklet, if the venue uses one
    "notebook",     # ruled or unruled exercise book
    "pen",
    "pencil",
    "eraser",
    "ruler",
    "calculator",
    "cell_phone",
    "laptop",
    "tablet",
    "headphones",
    "sheet_of_paper_stack",  # a stack of loose paper / printed pack
    "earbuds",               # appended (id 13); wired to P26 Earbuds
    # Appended after the original 14-class contract.  Appending keeps old
    # weights and labels valid; reordering would silently reinterpret every
    # existing class id.
    "exam_booklet",          # venue-issued answer booklet (authorized)
    "loose_scratch_paper",   # loose notes/scratch paper (unauthorized on desk)
]

# Documents the venue issues to every candidate.  These are expected on the
# desk and must never escalate on their own.
AUTHORIZED_DOCUMENT_CLASSES: frozenset[str] = frozenset({
    "exam_paper", "answer_sheet", "exam_booklet",
})

# Documents that are only evidence when they intersect the candidate's
# designated desk space: a notebook in a bag on the floor is not note-peeking,
# one open on the desk is.
UNAUTHORIZED_DOCUMENT_CLASSES: frozenset[str] = frozenset({
    "notebook", "loose_scratch_paper", "sheet_of_paper_stack",
})

# Objects that are individually innocent but become evidence only in
# combination with a spatial relation.  Kept as a named group because the
# escalation logic should treat these differently from a phone, which is
# cheating on its own.
AMBIENT_CLASSES: frozenset[str] = frozenset({
    "exam_paper", "answer_sheet", "exam_booklet", "notebook",
    "loose_scratch_paper", "sheet_of_paper_stack",
})

# Classes that indicate a device or a prohibited aid regardless of context.
# Kept for the escalation rules, not for detection.
DEVICE_CLASSES: frozenset[str] = frozenset({
    "cell_phone", "laptop", "tablet", "headphones", "earbuds",
})


def class_id(name: str) -> int:
    """Return the model class id for *name*.

    Raises KeyError for an unknown name, so a typo in a data-config file
    fails loudly instead of silently training on the wrong label.
    """
    return DESK_CLASSES.index(name)


def name_for_id(idx: int) -> str:
    """Return the class name for a model class id."""
    return DESK_CLASSES[idx]


def is_known(name: str) -> bool:
    return name in DESK_CLASSES


if __name__ == "__main__":
    print(f"{len(DESK_CLASSES)} desk classes")
    for i, name in enumerate(DESK_CLASSES):
        print(f"  {i:>2}  {name}")
    print()
    print(f"ambient (needs context): {sorted(AMBIENT_CLASSES)}")
    print(f"devices (context-free)  : {sorted(DEVICE_CLASSES)}")
