from collections import defaultdict

def build_class_to_patch_map(test_labels: dict[str, int]) -> dict[int, list[str]]:
  '''
  Builds a mapping from class IDs to lists of patch IDs.
  Args:
    test_labels (dict[str, int]): Mapping from patch ID to class ID.
  Returns:
    dict[int, list[str]]: Mapping from class ID to list of patch IDs.
  '''
  class_to_patch_ids = defaultdict(list)
  for patch_id, class_id in test_labels.items():
    class_to_patch_ids[class_id].append(patch_id)
  return dict(class_to_patch_ids)

def build_class_names_from_partition(
    test_partition: dict[int, list[str]]
) -> dict[int, str]:
    """
    Extracts class names (e.g., authors names) from filenames in a partition
    dictionary.
    Assumes filenames follow the pattern: 'authorname_test_patch_XXX.npy'
    Args:
      - test_partition (dict[int, list[str]]): Mapping from class ID to patch
      filenames.
    Returns:
      - dict[int, str]: Mapping from class ID to extracted author name.
    """
    class_names = {}
    for class_id, patch_ids in test_partition.items():
        first_patch = patch_ids[0]

        author_name = first_patch.split("_test")[0]
        class_names[class_id] = author_name
    return class_names

def group_patch_ids_by_class(
    patch_ids: list[str],
    labels: dict[str, int]
) -> dict[int, list[str]]:
  '''
  Groups a list of patch IDs by their class label.
  Args:
    - patch_ids (list[str]): List of patch IDs to group.
    - labels (dict[str, int]): Mapping from patch ID to class ID.
  Returns:
    dict[int, list[str]]: Mapping from class ID to list of patch IDs.
  '''
  grouped = defaultdict(list)
  for pid in patch_ids:
    grouped[labels[pid]].append(pid)
  return dict(grouped)