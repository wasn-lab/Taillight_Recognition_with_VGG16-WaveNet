import re

__BAG_NAME_RGX = re.compile(
    r".+_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})_[\d]+.bag")


def get_bag_yymmdd(bag):
    """Return the 8-char {year}{month}{day} string encoded in |bag|."""
    match = __BAG_NAME_RGX.search(bag)
    if not match:
        return "999999"
    year = match.expand(r"\g<year>")
    month = match.expand(r"\g<month>")
    day = match.expand(r"\g<day>")
    return year + month + day
