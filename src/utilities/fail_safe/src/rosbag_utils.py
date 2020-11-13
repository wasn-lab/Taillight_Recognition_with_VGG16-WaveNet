import re
import logging

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


def get_bag_timestamp_in_dict(bag):
    """Return the dict of timestamp info encoded in |bag|."""
    match = __BAG_NAME_RGX.search(bag)
    ret = {"year": "9999", "month": "99", "day": "99",
           "hour": "99", "minute": "99", "second": "99"}
    if not match:
        logging.warn("Cannot parse timestamp for %s", bag)
        return ret
    for key in ["year", "month", "day", "hour", "minute", "second"]:
        ret[key] = match.expand(r"\g<" + key + r">")
    return ret
