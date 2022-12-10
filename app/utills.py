import datetime
import calendar

def get_utc_time():
    date = datetime.datetime.utcnow()
    utc_time = calendar.timegm(date.utctimetuple())

    return utc_time