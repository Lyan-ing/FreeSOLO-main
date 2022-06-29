from detectron2.data.catalog import DatasetCatalog, MetadataCatalog


def get_detection_dataset_dicts(names, filter_empty=True, min_keypoints=0, proposal_files=None):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    # if proposal_files is not None:
    #     assert len(names) == len(proposal_files)
    #     # load precomputed proposals from proposal files
    #     dataset_dicts = [
    #         load_proposals_into_dataset(dataset_i_dicts, proposal_file)
    #         for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
    #     ]
    #
    # dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    #
    # has_instances = "annotations" in dataset_dicts[0]
    # if filter_empty and has_instances:
    #     dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    # if min_keypoints > 0 and has_instances:
    #     dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)
    #
    # if has_instances:
    #     try:
    #         class_names = MetadataCatalog.get(names[0]).thing_classes
    #         check_metadata_consistency("thing_classes", names)
    #         print_instances_class_histogram(dataset_dicts, class_names)
    #     except AttributeError:  # class names are not available for this dataset
    #         pass
    #
    # assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    # return dataset_dicts


get_detection_dataset_dicts('coco_2017_train_unlabeled_densecl_r101')

# test only push no commit
