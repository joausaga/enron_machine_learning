#!/usr/bin/python


# Code based on the quiz: Visualizing your new feature
# of the Feature Selection lesson
def compute_fraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0
    else:
        poi_messages = float(int(poi_messages))
        all_messages = float(int(all_messages))
        if all_messages > 0:
            fraction = 0.
            fraction = poi_messages/all_messages
            return fraction
        else:
            return 0