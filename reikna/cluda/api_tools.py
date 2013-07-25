import re


def _name_matches_masks(name, includes, excludes):
    if len(includes) > 0:
        for include in includes:
            if re.search(include, name):
                break
        else:
            return False

    if len(excludes) > 0:
        for exclude in excludes:
            if re.search(exclude, name):
                return False

    return True


def find_devices(
        api,
        include_devices=None, exclude_devices=None,
        include_platforms=None, exclude_platforms=None,
        include_duplicate_devices=True,
        include_pure_only=False):
    """
    Find platforms and devices meeting certain criteria.

    :param api: a CLUDA API object.
    :param include_devices: a list of masks for a device name
        which will be used to pick devices to include in the result.
    :param exclude_devices: a list of masks for a device name
        which will be used to pick devices to exclude from the result.
    :param include_platforms: a list of masks for a platform name
        which will be used to pick platforms to include in the result.
    :param exclude_platforms: a list of masks for a platform name
        which will be used to pick platforms to exclude in the result.
    :param include_duplicate_devices: if ``False``, will only include a single device
        from the several with the same name available on a platform.
    :param include_pure_only: if ``True``, will include devices with maximum group size equal to 1.
    :returns: a dictionary with found platform numbers as keys,
        and lists of device numbers as values.
    """

    if include_devices is None:
        include_devices = []
    if exclude_devices is None:
        exclude_devices = []
    if include_platforms is None:
        include_platforms = []
    if exclude_platforms is None:
        exclude_platforms = []

    devices = {}

    for pnum, platform in enumerate(api.get_platforms()):

        seen_devices = set()

        if not _name_matches_masks(platform.name, include_platforms, exclude_platforms):
            continue

        for dnum, device in enumerate(platform.get_devices()):
            if not _name_matches_masks(device.name, include_devices, exclude_devices):
                continue

            if not include_duplicate_devices and device.name in seen_devices:
                continue

            params = api.DeviceParameters(device)
            if not include_pure_only and params.max_work_group_size == 1:
                continue

            seen_devices.add(device.name)

            if pnum not in devices:
                devices[pnum] = []

            devices[pnum].append(dnum)

    return devices
